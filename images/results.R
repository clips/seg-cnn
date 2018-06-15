library(ggplot2)
file1 = "../result/results_test_trp"
f1 = read.table(file1, header=TRUE, sep=" ", na.strings="NA", dec=".", strip.white=TRUE)
my= ggplot() +
geom_errorbar(data=f1, mapping=aes(x=model, ymin=upper, ymax=lower), width=0.2, size=1, color="black") +
geom_point(data=f1, mapping=aes(x=model, y=F1), size=4, shape=21, fill="white") +
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

ggsave("results_test_trp.pdf", plot=my, width = 7, height = 5)
