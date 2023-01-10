import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import pdb
import glob

epoch_sgd = 50
batch_size = 16
momentum = True
lr = 1e-3
img_d = ''
img_n = []
img_size = 224


def show_data(base_dir):
    global img_d
    global img_n
    img_d = base_dir
    plt.figure(figsize=(15,8))
    img_list = os.listdir(img_d + 'valid/')
    img_list = sorted(img_list, key=lambda x: x[0:6])
    
    cnt = 0
    for name in img_list[::-1]:
        if len(name) == 19:
            img = Image.open(img_d + 'valid/' + name)
            img_n.append(name)
            plt.subplot(1,6,cnt+1)
            plt.imshow(img)
            cnt += 1
            if cnt == 6:
                break



def prepare_data(sub, base_dir='DATASET/'):
    test_cnt = 500
    if sub != 'test':
        age = np.loadtxt(base_dir + sub + ".txt", delimiter=',')
    else:
        age = None
    H = np.load(base_dir + 'feature_' + sub + '.npy')
    return age, H


def evaluate(w, b, age, feature):
    pred = predict(w,b,feature)
    loss = np.abs(pred - age).mean()
    
    plt.figure(figsize=(15,8))
    for i in range(6):
        img = cv2.imread(img_d + 'valid/' + img_n[i])
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.putText(img, str(int(pred[::-1][i])), (0, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        img = cv2.putText(img, str(int(age[::-1][i])), (180, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        plt.subplot(1,6,i+1)
        plt.imshow(img)
    
    return loss

def show_results(preds, gt):
    plt.figure(figsize=(15,8))
    img_size = 225
    for i in range(6):
        img = cv2.imread(img_d + 'valid/' + img_n[i])
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.putText(img, str(int(preds[::-1][i])), (0, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        img = cv2.putText(img, str(int(gt[::-1][i])), (180, 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        plt.subplot(1,6,i+1)
        plt.imshow(img)


def test(w, b, feature, filename='results.txt'):
    pred = predict(w,b,feature)
    np.savetxt(filename, pred, delimiter=',')
    return pred

def test_cel(model, loader, filename):
    model.eval()
    preds = []
    for i, (y,x) in enumerate(loader):
        x= x.cuda().float()
        outputs = model(x)
        preds.append(F.softmax(outputs, dim=-1).cpu().detach().numpy())

    preds = np.concatenate(preds, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (preds * ages).sum(axis=-1)
    np.savetxt(filename, ave_preds, delimiter=',')
    return ave_preds


def predict(w,b, feature):
    pred = np.dot(feature, w) + b.reshape(1,-1)
    exp = np.exp(pred)
    prob = exp / np.sum(exp, axis=-1)[:, None]
    pred = np.dot(prob,np.arange(0,101))
    return pred


#############################################################################################
# Visualize function for linear perceptron
#############################################################################################
def evaluate_sgd_with_hidden_layer(w, b, age, feature):
    x = np.dot(feature, w[0]) + b[0].T
    x = np.maximum(x, 0)
    x = np.dot(x, w[1]) + b[1].T
    loss = np.power(x - age,2).mean()
    
    plt.figure(figsize=(15,8))
    for i in range(6):
        img = cv2.imread(img_d + 'valid/' + img_n[i])
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.putText(img, str(int(x[::-1][i])), (0, 25),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        img = cv2.putText(img, str(int(age[::-1][i])), (180, 25),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        plt.subplot(1,6,i+1)
        plt.imshow(img)
        
    return loss


def test_sgd_with_hidden_layer(w, b, feature, filename='sgd_hidden.txt'):
    x = np.dot(feature, w[0]) + b[0].T
    x = np.maximum(x, 0)
    x = np.dot(x, w[1]) + b[1].T
    np.savetxt(filename, x, delimiter=',')
    return x


#############################################################################################
# Visualize function for linear perceptron
#############################################################################################
def visualize_results(images, preds, labels, alpha):
    num_samples = images.shape[0]
    y_hat=np.ones_like(preds, dtype=np.int16)
    y_hat[np.less_equal(preds, np.zeros_like(preds))]=-1
    fig, axes = plt.subplots(1, num_samples)
    items = sorted([[index, value] for index, value in enumerate(preds)], key=lambda x:x[1])
    sorted_idx = [item[0] for item in items]
    images = images[sorted_idx]
    labels = labels[sorted_idx]
    if alpha:
        alpha  = alpha[sorted_idx]
    for i in range(images.shape[0]):
        axes[i].axis('off')
        if alpha:
            axes[i].set_title('{}'.format(alpha[i][0]), fontdict={'fontsize':30})
        else:
            axes[i].set_title('{}'.format(labels[i][0]), fontdict={'fontsize':30})
        image = images[i].copy()
        if labels[i] == 1:
            image = image[...,::-1]
        axes[i].imshow(image)
    plt.show()


"""
def load_image(file_names, size=(20, 20)):
    img_arr, lab_arr, reduced = [], [], []
    for file_name in file_names:
        _, name_type, name = file_name.split(os.sep)
        img, lab = cv2.imread(file_name), -1
        reduced.append(cv2.resize(img, size, interpolation = cv2.INTER_CUBIC))
        if name_type == 'train_smile': 
            lab = 1
        img_arr.append(img)
        lab_arr.append(lab)
    return np.stack(reduced), np.stack(img_arr), np.stack(lab_arr)[...,None]
"""
def load_image(file_names, size=(20, 20)):
    img_arr, lab_arr, reduced = [], [], []
    for file_name in file_names:
        tokens = file_name.split(os.sep)
        if (len(tokens) >= 2):
            name_type = tokens[-2]

        img, lab = cv2.imread(file_name), -1
        reduced.append(cv2.resize(img, size, interpolation = cv2.INTER_CUBIC))
        if name_type == 'train_smile': 
            lab = 1
        img_arr.append(img)
        lab_arr.append(lab)
    return np.stack(reduced), np.stack(img_arr), np.stack(lab_arr)[...,None]


def test_on_emoji_dataset(p_class):
    file_names = glob.glob('data/*/*.*')
    reduced, images, labels = load_image(file_names)
    p = p_class(reduced.reshape(reduced.shape[0], -1), labels)
    for i in range(100): 
        p.update()
        preds, y_hat = p.predict()
        visualize_results(images, preds, labels, p.alpha)