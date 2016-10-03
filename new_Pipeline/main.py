from pipeline import *
from p1_VGG import *
import pickle
import matplotlib.pyplot as plt

#Test = pickle.load( open( "validation_set.p", "rb" ) )
Test = pickle.load( open( "/home/samiran/TRI/Build_Dataset/testing_links.p", "rb" ) )


#Unset this to turn off visualization

Vis=1

#Set this to see the final image full screen
full_image = 1
image_count = 10
g = open("time.csv","w")
g.write("count,avg_latency,precision,recall\n")
g.close

f=open("stats.csv","w")
f.write("")
f.close
res=open("stats.csv","a+")
g=open("time.csv","a+")


if(full_image == 0):
    plt.ion()
    f, axarr = plt.subplots(2,2)

ids,model_heatmap = initialize_vgg() 

def create_model():
    nb_classes = 2

    # input image dimensions
    img_rows, img_cols = 64, 64
    img_channels = 3
    # Create the model
    model = Sequential()
    model.add(Convolution2D(128, 3, 3, input_shape=(3, 64, 64), activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
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

overall_lights = 0
overall_true_pos = 0
overall_false_pos = 0

total_latency = 0
avg_latency = 0

count = 0

red_density = 0.4
Total_images=len(Test.keys())

it=1
for i in Test.keys():
    print "Processing image ",it,"/",Total_images
    it+=1
    print("")
    total_time, original_image,heatmap,thresh,final_annotated_image,total_lights,true_pos,false_pos=run_pipeline(i,Test[i],ids,model_heatmap,model,red_density)
    overall_lights+=total_lights
    overall_true_pos+=true_pos
    overall_false_pos+=false_pos
    if(count > 0):
        total_latency += total_time

    #initial latency for drawing plot is a lot
    if(count > 1):
        total_latency += total_time
        print "Frame Latency: ", total_time, " s"
        avg_latency = (total_latency*1.0)/(count-1)
        precision = (overall_true_pos*1.0)/(overall_true_pos+overall_false_pos)
        recall = (overall_true_pos*1.0)/overall_lights
        print "Avg. Latency:", avg_latency
        print"Avg. Precision:",precision, " Avg. Recall:", recall
        g.write(str(count) +","+ str(avg_latency)+","+str(precision)+","+str(recall)+"\n")
    
    if(full_image == 0) and Vis!=0:
        # Two subplots, the axes array is 1-d
        axarr[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), cmap=plt.gray())
        axarr[0, 0].set_title('Original')
        axarr[0, 1].imshow(heatmap, cmap=plt.gray())
        axarr[0, 1].set_title('Heatmap')
        axarr[1, 0].imshow(thresh, cmap=plt.gray())
        axarr[1, 0].set_title('Thresholded')
        axarr[1, 1].imshow(final_annotated_image, cmap=plt.gray())
        axarr[1, 1].set_title('Final Annotations')
        plt.pause(2)

    elif Vis!=0:
         plt.imshow(final_annotated_image)
         plt.pause(5)    
    count += 1
    #if(count > image_count):
    #    break

avg_latency = (total_latency*1.0)/(count-1)
precision = (overall_true_pos*1.0)/(overall_true_pos+overall_false_pos)
recall = (overall_true_pos*1.0)/overall_lights
f1_score = 0
if(precision > 0 or recall > 0):
    f1_score = (2 * precision * recall)/(precision + recall)

print("")
print "Avg. Latency:", avg_latency
print"Avg. Precision:",precision, " Avg. Recall:", recall, "F1 Score:", f1_score


res.write("red_density,precision,recall,F1_score,TP,FP,tot_lights\n")
res.write(str(red_density) +","+ str(precision)+","+str(recall)+","+str(f1_score)+
    ","+str(overall_true_pos) +","+ str(overall_false_pos)+","+str(overall_lights)+"\n")

res.close()
g.close()
