# World Bank Data on Economics and Arms 
# What should the model start with? 

library(wbstats)
library(countrycode)
library(readr)
library(dplyr)
library(rio)

## All states in lates list from COW
states <- read_csv("states2016.csv") %>%
    filter(endyear == 2016) %>%
    select(c(stateabb, ccode, statenme))

## World bank government spending
expenditure <- wb(
    indicator = "NE.CON.GOVT.CN",
    startdate = 2017,
    enddate = 2017
    )

## Merge
## Create common id variable for merging
expenditure$stateabb <- countrycode(
    expenditure$iso3c, "iso3c", "cowc"
    )

## Remove extra columns
expenditure <- select(expenditure, c(stateabb, value))
colnames(expenditure) <- c("stateabb", "gov_exp")

## Merge
state_exp <- left_join(states, expenditure)


## World bank gdp
gdp <- wb(
    indicator = "NY.GDP.MKTP.CN",
    startdate = 2017,
    enddate = 2017
    )

## Merge
## Create common id variable for merging
gdp$stateabb <- countrycode(
    gdp$iso3c, "iso3c", "cowc"
    )

## Remove extra columns
gdp <- select(gdp, c(stateabb, value))
colnames(gdp) <- c("stateabb", "gdp")

## Merge
state_exp <- left_join(state_exp, gdp)

## Military spending -- SIPRI
milex <- wb(
    indicator = "MS.MIL.XPND.CN",
    startdate = 2017,
    enddate = 2017
    )

## Merge
## Create common id variable for merging
milex$stateabb <- countrycode(
    milex$iso3c, "iso3c", "cowc"
    )

## Remove extra columns
milex <- select(milex, c(stateabb, value))
colnames(milex) <- c("stateabb", "milex")

## Merge
state_exp <- left_join(state_exp, milex)

## Export
export(state_exp, file = "real_dat.rds")