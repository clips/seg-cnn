require(ggplot2)
file2 = "learningcurve"
f2 = read.table(file2, header=TRUE, sep=" ", na.strings="NA", dec=".", strip.white=TRUE)

my2 = ggplot(f2, aes(f2$size, f2$f1)) + geom_point() + geom_line() +
#scale_x_continuous("N of tokens in passage", breaks = c(0,500,1000,1500,2000, 2500, 3000, 6000), labels= c(0,500,1000,1500,2000,2500, 3000, 6000)) +
#scale_y_continuous("Density", breaks=c(0.0, 2e-04, 4e-04, 6e-04), labels=c(0, "2e-4", "4e-4", "6e-4")) +
   theme_bw() +
   theme(plot.title = element_text(size=12),
	axis.text.x = element_text(size=12, angle=90, vjust=0.5),
	axis.text.y = element_text(size=12, angle=90, hjust=0.5),
	axis.title.x = element_text(size=12),
	axis.title.y = element_text(size=12, angle=90),
	panel.grid.minor = element_blank(),
	legend.justification=c(0,1), #left/right, bottom/top
	legend.position=c(0,1),
	legend.title = element_blank(),
	legend.text = element_text(size=12),
	legend.key.size = unit(1.4, "lines"),
	legend.key = element_rect(colour=NA),
	panel.border = element_blank())

ggsave("learningcurve.pdf", plot=my2, width = 7, height = 5)
