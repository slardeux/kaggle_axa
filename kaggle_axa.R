setwd("~/data_project/axa_kaggle")
#############################################################################################################
## load the csv create the binary
###########################################################################################################

library(data.table)
# Apply trip ID to new third column in data frame
fread.and.modify <- function(file.number, driver) {
  tmp <- fread(paste0("drivers/",driver,"/",file.number,".csv"), header=T, sep=",")
  tmp[, tripID:=file.number]
  return(tmp)
}
# Pull down list of driver directories and create a home for binaries
driverlist <- list.files("./drivers/")
dir.create("./data/", showWarnings = TRUE, recursive = FALSE, mode = "0777")
# Loop through the driver list and use rbindlist to bind data from
# x, and y columns to the specific driver data frame
library(doParallel)
library(foreach)
registerDoParallel(cores=3)
foreach(i = 1:length(driverlist)) %dopar% {
  onedriver <- driverlist[i]
  drives <- rbindlist(lapply(1:200, fread.and.modify, onedriver))
  save(drives, file = paste('./data/DriverData',onedriver, sep=''))
}

################################################################################################################
## create variable
###############################################################################################################
setwd("~/data_project/axa_kaggle")
library(plyr)
library(dplyr)
library(doParallel)

dir.create("./data_transf/", showWarnings = TRUE, recursive = FALSE)

turn_trip.f <- function(trip){
  # turn the trip by deciding it begin to go up and right (y>0 and x>0)
  if(trip$x[which.max(abs(trip$x) > 0.1)] < 0){trip$x = -trip$x}
  if(trip$y[which.max(abs(trip$y) > 0.1)] < 0){trip$y = -trip$y}
  return(trip)
}

calculate_trip_value.f <- function(trip){
  #this function calculate vel, speed accel, put a flag if data are bad and put a flag when car is moving
  trip <- trip %>% mutate(dist = sqrt((x-lag(x))^2 + (y - lag(y))^2), velx = x -lag(x), vely = y - lag(y), speed = sqrt(velx^2 + vely^2), accel = speed - lag(speed), jerk = accel - lag(accel), angle = atan2((y - lag(y)), (x - lag(x)))*180/pi)
  
  ##if missing points aka 'jump'
  trip$jump <- ifelse(abs(trip$accel) > 100, TRUE, FALSE)
  
  #boolean for when the car turn
  trip$turnpos1 <- ifelse((trip$angle - lag(trip$angle)) > 45, TRUE, FALSE)
  trip$turneg1 <- ifelse((trip$angle - lag(trip$angle)) < - 45, TRUE, FALSE)
  trip$turnpos2 <- ifelse((trip$angle - lag(trip$angle, 2)) > 45, TRUE, FALSE)
  trip$turneg2 <- ifelse((trip$angle - lag(trip$angle, 2)) < - 45, TRUE, FALSE)
  trip$turnpos3 <- ifelse((trip$angle - lag(trip$angle, 3)) > 45, TRUE, FALSE)
  trip$turneg3 <- ifelse((trip$angle - lag(trip$angle, 3)) < - 45, TRUE, FALSE)
  return(trip)
}

expand_data.f <- function(x, turn_trip.f, calculate_trip_value.f){
  #this function load the data called the previous function and save the data 
  load(paste0('data/',x))
  tmp <- lapply(split(drives, drives$tripID), turn_trip.f)  
  drives <- lapply(tmp, calculate_trip_value.f)   
  save(drives, file = paste('data_transf/', x, sep = ''))
}

todo <- list.files('data')
registerDoParallel(cores=6)
foreach(i=1:length(todo)) %dopar% {
  expand_data.f(todo[i], turn_trip.f, calculate_trip_value.f)  
}

###############################################################################################################
## summarise the trip data for each trip
###############################################################################################################
setwd("~/data_project/axa_kaggle")

library(plyr)
library(dplyr)
library(reshape2)
library(tidyr)
library(doParallel)

dir.create("./data_summary/", showWarnings = TRUE, recursive = FALSE)

trip_feature.f <- function(trip){
  #this function summarise the variable value for each trip and create new variable for each trip
  len <- dim(trip)[1]
  dist <- sum(trip$dist, na.rm = TRUE)
  trip <- as.data.frame(trip)
  tmp <- trip %>% filter(!jump) %>% select(velx, vely, speed, accel, jerk, angle) %>% filter(!is.na(jerk))
  res <- t(sapply(tmp, function(x) each(mean, sd, quantile)(x)))
  res <- data.frame(param = rownames(res) , res)
  rownames(res) <- NULL
  meltres <-  melt(res, id = 'param')
  res <- data.frame(var = paste(meltres$param, meltres$variable, sep = '-'), value = meltres$value)
  res$var <- as.character(res$var)
  turn <- as.data.frame(colSums(trip %>% select(turnpos1:turneg3) %>% filter(!is.na(turneg3))))
  turn <- data.frame(var = rownames(turn), value = turn[,1])
  res <- rbind(res, turn)
  angletodest <- with(trip, atan2(y[len] - y[1], x[len]-x[1]))
  angle_change <- sum(tmp$angle)
  tot_angle <- sum(abs(tmp$angle))
  turn_efficiency <- angle_change/tot_angle
  direct_dist <- with(trip, sqrt((x[len] - x[1])^2 + (y[len] - y[1])^2))
  path_efficiency <- direct_dist/dist
  res <- rbind(res, c('len', len))
  res <- rbind(res, c('dist', dist))
  res <- rbind(res, c('angletodest', angletodest))
  res <- rbind(res, c('angle_change', angle_change))
  res <- rbind(res, c('turn_efficiency', turn_efficiency))
  res <- rbind(res, c('tot_angle', tot_angle))
  res <- rbind(res, c('direct_dist', direct_dist))
  res <- rbind(res, c('path_efficiency', path_efficiency))
  
  res <- data.frame(tripID = unique(trip$tripID), res)
  res <- spread(res, var, value)
  return(res)
}

find_feature.f <- function(x, trip_feature.f){
  load(paste0('data_transf/',x))
  res <- lapply(drives, trip_feature.f)
  drives <- do.call(rbind, res)
  save(drives, file = paste('data_summary/', x, sep = ''))
}

todo <- list.files('data_transf')
registerDoParallel(cores=6)
foreach(i=1:length(dir())) %dopar% {
  find_feature.f(todo[i], trip_feature.f)  
}

################################################################################################################
## lasso
################################################################################################################
setwd("~/data_project/axa_kaggle")
drivers = list.files("data_summary")
randomDrivers = sample(drivers, size = 5)
library(dplyr)
library(reshape2)
library(tidyr)
library(glmnet)
library(doParallel)

registerDoParallel(cores=3)
refData<-foreach(driver=randomDrivers, .combine=rbind) %dopar% {
  load(paste0("data_summary/",driver))
  spreadD <- drives %>% spread(var, value)
}
target <- 1
refData$target <-target  

target <- 0
registerDoParallel(cores=6)
submission<-foreach(driver=drivers,.combine=rbind) %dopar% {
  
  ## need to grab the numeric driverID from the file name
  driverID <- gsub(driver,pattern="DriverData",replacement="")
  load(paste0("data_summary/",driver))
  print(driver)
  currentData <- drives
  currentData$target <- target
  
  train <- as.data.frame(rbind(refData,currentData))
  train <- apply(train,2, as.numeric)
  train[which(is.na(train))] = 0
  currentData <- apply(currentData ,2, as.numeric)
  x <- train[,2:93]
  y <- as.factor(train[,94])
  grid =10^ seq (10 , -2 , length =100)
  lasso.mod <- glmnet(x, y, family = 'binomial',alpha =1 , lambda = grid )
  set.seed (1)
  cv.out = cv.glmnet (train[,2:93], train[,94], alpha =1)
  bestlam = cv.out$lambda.min
  p = predict(lasso.mod , s = bestlam , newx = currentData[,2:93], type = 'response')
  p = ifelse(p > 0.5, 1, 0)
  labels <- sapply(1:200, function(x) paste0(driverID,'_', x))
  data.frame(labels, p)
}

colnames(submission) = c("driver_trip","prob")
write.csv(submission, "submission.csv", row.names=F, quote=F)


########################################################################################################
## lasso with matching trip
########################################################################################################
setwd("~/data_project/axa_kaggle")
load('clusters-0.5')
drivers <- list.files("data_summary")
randomDriver = sample(drivers, size = 1)
library(dplyr)
library(reshape2)
library(tidyr)
library(glmnet)
library(doParallel)
library(foreach)
registerDoParallel(cores=3)

load(paste0("data_summary/",randomDriver))
refData <- drives %>% spread(var, value)
refData$target <-0

registerDoParallel(cores=6)
submission<-foreach(driver=drivers,.combine=rbind) %dopar% {
  
  # in case the driver is also in the refdata, we want to change the ref data
  if(randomDriver == driver){
    newdriver <- sample(drivers, size = 1)
    while(newdriver == driver){
      newdriver <- sample(drivers, size = 1)
    }
    load(paste0("data_summary/",newdriver))
    refData  <- drives %>% spread(var, value)
    refData$target <- 0
  }
   
  ## need to grab the numeric driverID from the file name
  driverID <- gsub(driver,pattern="DriverData",replacement="")
  load(paste0("data_summary/",driver))
  print(driver)
  
  #get the matching trip:
  group <- grouped %>% filter(driver == as.numeric(driverID))
  matching <- as.numeric(unlist(strsplit(big$matches, ', '))) 
  
  currentData <- drives %>% spread(var, value)
  if(length(matching) == 0){
    matchData <- currentData
    test <- currentData
  }else{
    matchData <- currentData %>% filter(tripID %in% matching)
    test <- currentData %>% filter(!(tripID %in% matching))
  }
  
  matchData$target <- 1
  names(refData) <- names(matchData)
  train <- as.data.frame(rbind(refData,matchData))
  train <- apply(train,2, as.numeric)
  train[which(is.na(train))] = 0
  test <- apply(test ,2, as.numeric)
  x <- train[,2:57]
  y <- as.factor(train[,58])
  grid =10^ seq (10 , -2 , length =100)
  lasso.mod <- glmnet(x, y, family = 'binomial',alpha =1 , lambda = grid )
  set.seed (1)
  cv.out = cv.glmnet (train[,2:57], train[,58], alpha =1)
  bestlam = cv.out$lambda.min
  p = predict( lasso.mod , s = bestlam , newx = test[,2:57], type = 'class')
  resp <- data.frame(test[,1], as.numeric(p))
  names(resp) = c('tripID', 'target')
  resp <- rbind(resp, matchData[,c(1,58)])
  resp <- arrange(resp, tripID) %>% select(target)
  labels <- sapply(1:200, function(x) paste0(driverID,'_', x))  
  data.frame(labels, resp)
}

colnames(submission) = c("driver_trip","prob")
write.csv(submission, "submission.csv", row.names=F, quote=F)
























