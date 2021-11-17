from Explain_LISA.Explanations import Explanations

class ExplainLISA:
    def __init__(self,img,class_names,img_shape,model,img1,img2,scale=True,filter_radius=10) -> None:
        """
            img: local path of img to be explained
            class_names: the classes available as predictions for a given model
            img_shape: shape of the image accept by the neural network
            model: the model to be explained get from tf.keras.models.load_model("your model path")
            img1: local path background data point for produce explanations with SHAP
            img2: local path background data point for produce explanations with SHAP
            scale: for manual image scaling if scaling layer absent in the model to be explained 
            filter_radius: the pixel value of the radius of the High pass filter
        """
        
        self.img = self.saveLoadAndPrep(img,int(img_shape),scale)
        self.img1 = self.saveLoadAndPrep(img1,int(img_shape),scale)
        self.img2 = self.saveLoadAndPrep(img2,int(img_shape),scale)
        self.model = model
        self.class_names = class_names
        self.img_shape = int(img_shape)
        self.img_list = []
        self.results = []
        for i in [self.img1,self.img2,self.img]:
          self.img_list.append(i)
        self.filter_radius=filter_radius
        
    def getExplanations(self):
        self.ExplanationsObj=Explanations(self.img,self.model,list(self.class_names),self.img_list,int(self.img_shape),self.filter_radius)
        self.ExplanationsObj.callForMethods()
        

    def saveLoadAndPrep(self,img,img_shape,scale):
        import tensorflow as tf
        img = tf.io.read_file(img)
        
        # decode image into tensor
        img = tf.io.decode_image(img,channels=3) # hardcode for 3 channels to be compatible despite of the image type

        # resize the image
        img = tf.image.resize(img,[img_shape,img_shape])

        # Scale Y/N
        if scale:
          return img/255.
        else:
          return img

