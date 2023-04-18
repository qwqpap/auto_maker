# auto_maker

一种用于自动生成yolov5所需数据集的软件


## 你需要做什么
把目标图片放进以目标名字命名的文件夹
把目标名字文件夹放在image文件夹下面
确保每个类别的图片数量是一致的
把背景图片放在back文件夹
运行main.py
大概会是这样：


/image


	/image/class_a
	
	
		/image/class_a/anything.jpg
		
		
		......
		
		
	/image/class_b
	
	
	......
	
	
/back


	/back/back_image.jpg
	
	
	.....
