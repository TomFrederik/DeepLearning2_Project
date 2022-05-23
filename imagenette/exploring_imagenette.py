import pickle
import matplotlib.pyplot as plt

val_in = open("imagenet/data/mini-imagenet-cache-val.pkl", "rb")

val = pickle.load(val_in)
print(list(val.keys()))
Xval = val["image_data"]

print(Xval.shape)
print(list(val['class_dict'].keys()))

plt.imshow(Xval[val['class_dict']['n01855672'][0]])
plt.show()
plt.close()

plt.imshow(Xval[val['class_dict']['n01855672'][4]])
plt.show()
plt.close()

plt.imshow(Xval[val['class_dict']['n02091244'][0]])
plt.show()
plt.close()

plt.imshow(Xval[val['class_dict']['n02091244'][4]])
plt.show()
plt.close()

# Xval = Xval.reshape([20, 600, 84, 84, 3])