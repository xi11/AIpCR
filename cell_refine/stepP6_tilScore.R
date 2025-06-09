setwd('/Users/xiaoxipan/Library/CloudStorage/OneDrive-InsideMDAnderson/yuanlab/Projects/ARTEMIS/result')

library(dplyr)
library(tidyr)
library(tidyverse)
library(readxl)
library(stringr)
library(ggpubr) #to add p-val on the plot
library(gridExtra)
library(RColorBrewer)
library(ggsci)
library(ggplot2)
library(readr) # to read tsv
library(stringr)
library(caret)
library(car) #colinearity
library(rpart) #Recursive partitioning analysis
library(pROC) #AUC
library(rpart.plot)

#ai-til from segformerBRCA and finetuned on Artemis
til <- read.csv('xxx/artemis_lei/discovery/til/4_cell_class_segformerBRCAartemis/combined_scoreOther_segformerBRCAartemis.csv')
tme <- read_excel('xxx/artemis_lei/discovery/mit-b3-finetunedBRCA-Artemis-e60-lr00001-s512-20x512/discovery_post_tme_pix.xlsx')
til$FileName <- substr(til$FileName, 1, nchar(til$FileName) - 4)
colnames(til)[colnames(til) == "FileName"] <- "ID"
colnames(til)[colnames(til) == "X.l_stroma"] <- "lym_stroma"
colnames(til)[colnames(til) == "X.f_stroma"] <- "fib_stroma"
colnames(til)[colnames(til) == "X.o_stroma"] <- "oth_stroma"
colnames(til)[colnames(til) == "X.t_tumor"] <- "tum_tumor"
colnames(til)[colnames(til) == "X.l_tumor"] <- "lym_tumor"
colnames(til)[colnames(til) == "X.o_tumor"] <- "oth_tumor"

#to convert pix-ss1 to area-mm2, without inflam
#pix-ss1 *16*16 to area at 20x
#area at 20x then rescale to the raw res, i.e., x rescale factor, which is 0.44/5 or 
#area at 20x to um2
#area pix to area mm2
tme$tumor_mm2 <- tme$tumor_pix *16 *16 *0.44*0.44*10e-6
tme$stroma_mm2 <- tme$stroma_pix *16 *16 *0.44*0.44*10e-6

tme$ID <- substr(tme$ID, 1, nchar(tme$ID)-4)
til <- merge(til, tme, by='ID')
til$ai-stilr <- til$lym_stroma / (til$lym_stroma + til$fib_stroma) # remove, count of lym within stroma mask/count of fibroblast within stroma mask
til$ai-stil <- til$lym_stroma / til$stroma_mm2 # lym density in stroma

til$ai-tilr <- (til$lym_stroma + til$lym_tumor) / (til$lym_stroma + til$fib_stroma +til$lym_tumor + til$tum_tumor) 
til$ai-til <- (til$lym_stroma + til$lym_tumor) / (til$stroma_mm2 + til$tumor_mm2) # remove
write.csv(til, 'xxxxxx.csv', row.names = FALSE)
