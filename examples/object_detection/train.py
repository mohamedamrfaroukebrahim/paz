import paz

images_12, class_args_12, boxes_12 = paz.datasets.load("VOC2012", "trainval")
images_07, class_args_07, boxes_07 = paz.datasets.load("VOC2007", "trainval")
images_07, class_args_07, boxes_07 = paz.datasets.load("VOC2007", "test")
