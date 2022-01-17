library(ggplot2)
library(dplyr)

#https://towardsdatascience.com/transform-data-and-create-stacked-bar-chart-using-ggplot2-6f2b3c3d12d6
#http://www.sthda.com/english/wiki/ggplot2-barplot-easy-bar-graphs-in-r-software-using-ggplot2
#https://www.r-graph-gallery.com/line-chart-several-groups-ggplot2.html

data <- read.csv(file = 'qaoa_cu.csv')
data_distance <- read.csv(file = 'qaoa_distance.csv')
data_multiple_distance <- read.csv(file = 'qaoa_multiple_distance.csv')
data_multiple_p <- read.csv(file = 'qaoa_multiple_p.csv')
data_multiple_p_distance = read.csv(file = 'qaoa_multiple_p_distance.csv')

data <- data %>%  filter(iteration < 2500)
data %>%
  ggplot( aes(x=state, y=probability, group=iteration, color=mean)) +
  geom_line()


data %>%
  ggplot( aes(x=mean, y=distance, group=iteration, color=mean)) +
  geom_line()

data_distance %>% ggplot(aes(x=iteration, y=distance))+
  geom_line()

#data_multiple_distance <- data_multiple_distance %>%  filter(iteration < 1500)
data_multiple_distance %>%
  ggplot( aes(x=iteration, y=distance, group=instance, color=instance)) +
  geom_line()

# multiple p all domain function f(x)_n convergence uniform
data_multiple_p <- data_multiple_p %>%  filter(p == 1)
data_multiple_p %>%
  ggplot( aes(x=state, y=probability, group=p, color=p)) +
  geom_line()

#data_multiple_distance <- data_multiple_distance %>%  filter(iteration < 1500)
data_multiple_p_distance %>%
  ggplot( aes(x=p, y=distance, group=instance, color=instance)) +
  geom_line()
