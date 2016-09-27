from convnetskeras.convnets import preprocess_image_batch, convnet
from convnetskeras.imagenet_tool import synset_to_dfs_ids

def initialize_vgg():
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model_heatmap = convnet('vgg_19',weights_path="weights/vgg19_weights.h5", heatmap=True)
    model_heatmap.compile(optimizer=sgd, loss='mse')
    traffic_light_synset = "n06874185"
    ids = synset_to_dfs_ids(traffic_light_synset)
    return ids,model_heatmap



def generate_heatmap(image):
        im_heatmap = preprocess_image_batch([filename], color_mode="bgr")
        out_heatmap = model_heatmap.predict(im_heatmap)
        heatmap = out_heatmap[0,ids].sum(axis=0)
        
        # print("heatmap:",np.shape(heatmap)) 
        # print(heatmap)
        my_range = np.max(heatmap) - np.min(heatmap)
        heatmap = heatmap / my_range
        heatmap = heatmap * 255
        # print(heatmap)
    
        
        heatmap = cv2.resize(heatmap,(width,height))
        # cv2.imshow("heatmap1",heatmap)


        heatmap[heatmap < 128] = 0    # Black
        heatmap[heatmap >= 128] = 255 # White