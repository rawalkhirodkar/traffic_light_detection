from pipeline import *
from p1_VGG import *
import pickle
import matplotlib.pyplot as plt

Test = pickle.load( open( "validation_set.p", "rb" ) )

plt.ion()
f, axarr = plt.subplots(2,2)
ids,model_heatmap = initialize_vgg()

f=open("dump.txt","wb")

def create_model():
    nb_classes = 2

    # input image dimensions
    img_rows, img_cols = 32, 32
    img_channels = 3
    # Create the model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes, activation='softmax'))

    return model

tm=time.time()
print "Loading model....",

model = create_model()
model.load_weights("new_traffic_light_weights.h5")
print "Took ", time.time()-tm, "s"





Total=0
Rec=0
Fal=0
count=0

for i in Test.keys():
    total_time, original_image,heatmap,thresh,final_annotated_image,Total_Lights,Recognized,Falsepos=run_pipeline(i,Test[i],ids,model_heatmap,model)
    Total+=Total_Lights
    Rec+=Recognized
    Fal+=Falsepos
    count+=1
    print "Time taken to process the frame: ", total_time, " s"
    # Two subplots, the axes array is 1-d

    axarr[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), cmap=plt.gray())
    axarr[0, 0].set_title('Original')
    axarr[0, 1].imshow(heatmap, cmap=plt.gray())
    axarr[0, 1].set_title('Heatmap')
    axarr[1, 0].imshow(thresh, cmap=plt.gray())
    axarr[1, 0].set_title('Thresholded')
    axarr[1, 1].imshow(final_annotated_image, cmap=plt.gray())
    axarr[1, 1].set_title('Final Annotations')

    #plt.tight_layout() # optional
    #plt.show()

    plt.pause(0.05)

    print "Average Accuracy: ", Rec*1.0/(Total)
    print "Average Precision", Rec*1.0/(Rec+Fal)
    f.write(str(Rec/(Rec+Fal))+"   "+str(Rec/(Total))+"\n")



f.close()