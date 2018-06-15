library(ggplot2)
file1 = "../result/cm"
file2 = "../result/cm_baseline"
f1 = read.table(file1, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f2 = read.table(file2, header=TRUE, sep="\t", na.strings="NA", dec=".", strip.white=TRUE)
f_diff = data.frame(Gold=f1$Gold, System=f1$System, value=f1$value - f2$value)

cm = ggplot(data = f_diff, aes(x=System, y=Gold, fill=value)) +
  geom_tile(color="white") +
  #scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, limit = c(-1,1), space = "Lab", name="Pearson\nCorrelation") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", space = "Lab", name="Count difference") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1)) +
  coord_fixed() +
  geom_text(aes(System, Gold, label = value), color = "black", size = 4)

ggsave("cm_diff_example.pdf", plot=cm, width = 7, height = 5)
