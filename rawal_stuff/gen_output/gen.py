# from keras.optimizers import SGD
# from convnetskeras.convnets import preprocess_image_batch, convnet
# from convnetskeras.imagenet_tool import synset_to_dfs_ids



# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model_heatmap = convnet('vgg_19',weights_path="../model/weights/vgg19_weights.h5", heatmap=True)
# model_heatmap.compile(optimizer=sgd, loss='mse')
# traffic_light_synset = "n06874185"
# ids = synset_to_dfs_ids(traffic_light_synset)
# im_heatmap = preprocess_image_batch(['1.png'], color_mode="bgr")
# out_heatmap = model_heatmap.predict(im_heatmap)
# heatmap = out_heatmap[0,ids].sum(axis=0)

# import matplotlib.pyplot as plt
# plt.imsave("heatmap_1.png",heatmap)


from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
from convnetskeras.imagenet_tool import synset_to_dfs_ids

im = preprocess_image_batch(['dog.jpg'], color_mode="bgr")

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model = convnet('vgg_19',weights_path="../model/weights/vgg19_weights.h5", heatmap=True)
model.compile(optimizer=sgd, loss='mse')

out = model.predict(im)

s = "n02084071"
ids = synset_to_dfs_ids(s)
heatmap = out[0,ids].sum(axis=0)

# Then, we can get the image
import matplotlib.pyplot as plt
plt.imsave("heatmap_dog.png",heatmap)