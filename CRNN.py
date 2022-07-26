#========================================================
#             CRNN.py - CNN+RNN+CTC -based scene text recognition
#========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json, cv2, os, time
import matplotlib.pyplot as plt

# argparse is used to conveniently set our configurations
import argparse

from RNNCell import RNNCell
from GRUCell import GRUCell


# use dictionary to assign each character with a unique integer
# use reverse_dictionary to assign each integer with corresponding character
character_set = 'abcdefghijklmnopqrstuvwxyz0123456789'  # 26 letters and 10 digits
ctc_dictionary = dict()
ctc_reverse_dictionary = dict()
ctc_dictionary[0] = '-'  # for CTC, label 0 --> 'blank' symbol
for i, char in enumerate(character_set):
    ctc_dictionary[char] = i + 1
    ctc_reverse_dictionary[i + 1] = char


class ListDataset(Dataset):
    def __init__(self, im_dir, norm_height=32, training=False):
        '''
        :param im_dir: path to directory with images and ground-truth file
        :param norm_height: image normalization height
        '''

        # get image paths and labels from label file
        with open(os.path.join(im_dir, 'gt.txt'), 'r') as f:
            self.im_paths = []
            self.labels = []

            lines = f.readlines()
            for line in lines:
                line = line.strip()
                im_name, label = line.split(', ')
                self.im_paths.append(os.path.join(im_dir, im_name))
                self.labels.append(label)

        print('------ Finish loading labels for %d images from %s ------' % \
            (len(self.im_paths), im_dir))

        # function: resize a image to one with fixed height
        def resize_height_transform(image):
            height = image.size[1]
            width = image.size[0]
            scale = float(norm_height) / height
            new_height = norm_height
            new_width = int(scale * width)
            new_image = transforms.functional.resize(
                image, (new_height, new_width))
            return new_image

        # image normalization and transform numpy arrays into tensors
        if training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.0, 0.0, 0.0)], p=0.5),
                transforms.RandomApply([transforms.RandomAffine(degrees=10.0, translate=(0.02, 0.05), shear=10.0)], p=0.5),
                transforms.Lambda(resize_height_transform),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Lambda(resize_height_transform),
                transforms.ToTensor(),
            ])

        self.dictionary = ctc_dictionary

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # read an image
        im_path = self.im_paths[index]
        im = cv2.imread(im_path)  #call a cv2 function to read an image from the aforementioned im_path
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # image pre-processing
        im = self.transform(im)  #call the class function 'transform' defined previously to pre-process im
        # convert values to [-1, 1]
        im.sub_(0.5).div_(0.5)

        # get the label && covert string into integer sequence
        str_label = self.labels[index]
        int_label = []
        for char in str_label:
            char = char.lower()  # for letters, only consider lower case
            if char in self.dictionary:
                int_label.append(self.dictionary[char])
            else:
                continue

        return im, int_label


def dataLoader(im_dir, norm_height, batch_size, workers=0, max_timestep=30):
    '''
    :param im_dir: path to directory with images and ground-truth file
    :param norm_height: image normalization height
    :param batch_size: batch size
    :param workers: number of workers for loading data in multiple threads
    :return: a data loader
    '''

    # owing to the images in one batch have different widths,
    # so we pad all images to the maximum width to construct the batch
    def ctc_sequence_collate(batch):
        batch_size = len(batch)
        im_height = batch[0][0].shape[1]
        im_widths = [im.shape[2] for im, label in batch]
        target_lengths = [len(label) for im, label in batch]
        max_width = max(im_widths)

        padded_ims = torch.zeros(
            (batch_size, 1, im_height, max_width), dtype=batch[0][0].dtype)
        collapsed_labels = []
        for i, (im, label) in enumerate(batch):
            padded_ims[i, 0, :, :im_widths[i]] = batch[i][0]
            collapsed_labels.extend(label)
        
        collapsed_labels = torch.tensor(collapsed_labels, dtype=torch.int32)
        im_widths = torch.tensor(im_widths, dtype=torch.int32)
        target_lengths = torch.tensor(target_lengths, dtype=torch.int32)
        return padded_ims, collapsed_labels, im_widths, target_lengths

    dataset = ListDataset(im_dir, norm_height, training=True if im_dir.find('train') != -1 else False)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True if im_dir.find('train') != -1 else False,  # shuffle images only when training
                      num_workers=workers,
                      collate_fn=ctc_sequence_collate)


# ==== Part 2: network structure

# ==== 2.1: a simple network based on CNN and RNN
class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()

        # a convolutional network with ReLU activation function
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, 
                kernel_size=7, stride=2,  # with stride = 2, output_size = input_size x 1/2
                padding=3),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=16, out_channels=32, 
                kernel_size=3, stride=2,  # with stride = 2, output_size = input_size x 1/2
                padding=1),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32, out_channels=32, 
                kernel_size=3, stride=1,
                padding=1),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=32, out_channels=32, 
                kernel_size=3, stride=1,
                padding=1),
            nn.ReLU(),
        )

        # a pooling layer that reduce feature map's height to 1, but keep its width unchanged

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, None))

        # a one-layer RNN/GRU
        self.hidden_size = 32
        self.rnn_cell = RNNCell(
            input_size=32, 
            hidden_size=self.hidden_size)
        # self.rnn_cell = GRUCell(
        #     input_size=32, 
        #     hidden_size=self.hidden_size)

        # a fully connected layer (i.e. Linear layer) that make classification among all charactcers (including 'blank' symbol)

        self.fc = nn.Linear(in_features=32, out_features=37)  # 26 letters, 10 digits and 1 'blank' symbol

    def forward(self, x):
        '''
        :param x: input images with size [batch_size, 1, height, width]
        :return: output with size [batch_size, width / 4, 37], where 37 is number of characters (including "blank" symbol)
        '''

        cnn_output = self.cnn(x)  # output_size: [batch_size, 32, height / 4, width / 4], where 32 is output channel of CNN
        pool_output = self.pool(cnn_output)  # output_size: [batch_size, 32, 1, width / 4]

        # ------ IMPORTANT: process of RNN recurrent forwarding (from left to right) ------
        batch_size = pool_output.shape[0]
        hidden_size = self.hidden_size
        rnn_output = []
        # TODO: complete the RNN recurrent forwarding
        h_t = torch.zeros((batch_size, hidden_size), dtype=torch.float, device=x.device)  # RNN's hidden state at timestep 0, size: [batch_size, hidden_size]
        for i in range(pool_output.shape[3]): # take the size of the width dimension of the output CNN feature map (pool_output)
            x_t = pool_output[:,:,0,i]  # x_t: get rnn sequential input at timestep t from the output CNN feature map (pool_output), please refer to Page 1, Problem 3 in Task 1 in hw4_readme.pdf

            # Recurrent computation for updated hidden state h_t, taking x_t and h_t (which is actually h_{t-1} in RNN equations) as input
            h_t = self.rnn_cell(x_t, h_t)
            rnn_output.append(h_t)
        # ------------------------------------------------------------

        # stack all hidden states into one tensor
        rnn_output = torch.cat(
            [h_t.unsqueeze(1) for h_t in rnn_output], 
            dim=1)  # output_size: [batch_size, width / 4, 32]

        # output = self.fc(pool_output.squeeze().permute(0, 2, 1))
        output = self.fc(rnn_output)  # output_size: [batch_size, width / 4, 37]

        return output


# ==== 2.2: greedy ctc decoder
class GreedyCTCDecoder(object):
    def __init__(self):
        super(GreedyCTCDecoder, self).__init__()

    def decode(self, preds, pred_lengths):
        '''
        :param preds: model predictions with size [batch_size, steps, 37], where 37 is number of characters (including "blank" symbol)
        :return: list of decoded strings
        '''
        decoded_results = []
        for pred, length in zip(preds, pred_lengths):
            length = max(length, pred.shape[0])

            max_per_step = pred[:length, :].argmax(dim=1)  # take the most likely output at each step

            result = self.merge_repeat(max_per_step)
            result = self.remove_blank(result)
            result = self.int2char(result)
            decoded_results.append(result)

        return decoded_results

    def merge_repeat(self, sequence):
        result = []
        prev_s = None
        for s in sequence:
            if s != prev_s:
                result.append(s)
            prev_s = s
        return result

    def remove_blank(self, sequence):
        result = []
        for s in sequence:
            if s != 0:
                result.append(s)
        return result

    def int2char(self, sequence):
        result = []
        for s in sequence:
            result.append(ctc_reverse_dictionary[int(s)])
        return ''.join(result)


# ==== Part 3: training and validation

def train_val(train_im_dir='data/train',
              val_im_dir='data/train',
              norm_height=32, n_epochs=20, batch_size=4,
              lr=1e-4, momentum=0.9, weight_decay=0.0,
              valInterval=5, device='cpu'):
    '''
    The main training procedure
    ----------------------------
    :param train_im_dir: path to directory with training images and ground-truth file
    :param val_im_dir: path to directory with validation images and ground-truth file
    :param norm_height: image normalization height
    :param n_epochs: number of training epochs
    :param batch_size: training and validation batch size
    :param lr: learning rate
    :param momentum: momentum of the stochastic gradient descent optimizer
    :param valInterval: the frequency of validation, e.g., if valInterval = 5, then do validation after each 5 training epochs
    :param device: 'cpu' or 'cuda'
    '''

    # training and validation data loader
    trainloader = dataLoader(train_im_dir, norm_height, batch_size)
    valloader = dataLoader(val_im_dir, norm_height, batch_size)
  
    # construct the model
    model = CRNN()
    model = model.to(device)
    ctc_decoder = GreedyCTCDecoder()
    # define CTC loss function for sequence recognition task
    criterion = nn.CTCLoss(blank=0, reduction='mean')

    # optimizer
    optimizer = optim.RMSprop(
        model.parameters(), lr, 
        momentum=momentum, 
        weight_decay=weight_decay)

    # training & validation
    losses = []  # to save loss of each training epoch
    accuracies = []  # to save accuracy of each validation epoch

    for epoch in range(n_epochs):
        # make sure the model is in training mode
        model.train()

        total_loss = 0.
        start_time = time.time()
        for step, batch in enumerate(trainloader):  # get a batch of data

            # clear gradients in the optimizer
            optimizer.zero_grad()  #call a function

            # set data type and device
            ims, labels, im_widths, target_lengths = batch
            ims, labels, im_widths, target_lengths = \
                ims.to(device), labels.to(device), im_widths.to(device), target_lengths.to(device)
            valid_pred_lengths = torch.ceil(im_widths / 4.0).type(torch.int32)  # feature map width = 1/4 image width

            # run the model
            out = model(ims)

            # compute loss
            preds = nn.functional.log_softmax(
                out.permute(1, 0, 2), dim=2)
            
            with torch.backends.cudnn.flags(enabled=False):
                loss = criterion(preds, labels, valid_pred_lengths, target_lengths)

            # backward
            loss.backward()
            total_loss += loss.item()

            # update parameters
            optimizer.step()  #call a function

        # show average loss value
        avg_loss = total_loss / len(trainloader)
        losses.append(avg_loss)
        epoch_time = time.time() - start_time
        print('Epoch {:02d}: loss = {:.3f}, time={:.2f}s'.format(epoch + 1, avg_loss, epoch_time))

        # validation
        if (epoch + 1) % valInterval == 0:
            # make sure the model is in evaluation mode
            model.eval()  # call a function to enter evaluation mode

            n_correct = 0.  # number of images that are correctly classified
            n_ims = 0.  # number of all the images

            with torch.no_grad():  # we do not need to compute gradients during validation

                for batch in valloader:

                    # set data type and device
                    ims, labels, im_widths, target_lengths = batch
                    ims, labels, im_widths, target_lengths = \
                        ims.to(device), labels.to(device), im_widths.to(device), target_lengths.to(device)
                    valid_pred_lengths = torch.ceil(im_widths / 4.0).type(torch.int32)  # feature map width = 1/4 image width

                    # run the model
                    out = model(ims)
                    preds = nn.functional.softmax(out, dim=2)

                    # decode labels into strings
                    decoded_labels = []
                    i = 0
                    for tlength in target_lengths:
                        dl = ''.join(
                            [ctc_reverse_dictionary[int(s)] for s in labels[i:i + tlength]]
                        )
                        decoded_labels.append(dl)
                        i += tlength

                    # decode (transcribe) the model output into strings
                    decoded_results = ctc_decoder.decode(preds, valid_pred_lengths)  

                    for dl, dr in zip(decoded_labels, decoded_results):
                        display_info = 'label: {} ==> pred: {}'.format(dl, dr)
                        if dl == dr:
                            n_correct += 1
                        else:
                            display_info = '[x] ' + display_info
                        n_ims += 1

            # show prediction accuracy
            print('Epoch {:02d}: validation word accuracy = {:.1f}%'.format(epoch + 1, 100 * n_correct / n_ims))
            accuracies.append((epoch + 1, 1.0 * n_correct / n_ims))

            # save model parameters in a file
            model_save_path = 'ctc_saved_models/model_epoch{:02d}.pth'.format(epoch + 1)
            torch.save({'state_dict': model.state_dict()}, model_save_path)
            print('Model saved in {}\n'.format(model_save_path))

    # draw the loss and accuracy curve
    plot_loss_and_accuracies(losses, accuracies)


# ==== Part 4: test
def test(model_path, im_dir='data/test',
         norm_height=32, batch_size=4, device='cpu'):
    '''
    Test procedure
    ---------------
    :param model_path: path of the saved model
    :param im_dir: path to directory with images and ground-truth file
    :param norm_height: image normalization height
    :param batch_size: test batch size
    :param device: 'cpu' or 'cuda'
    '''

    # construct the model
    model = CRNN()
    model = model.to(device)
    ctc_decoder = GreedyCTCDecoder()

    # load parameters we saved in model_path
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print('[Info] Load model from {}'.format(model_path))

    model.eval()  # call a function to enter evaluation mode

    # test loader
    testloader = dataLoader(im_dir, norm_height, batch_size)

    # run the test
    n_correct = 0.
    n_ims = 0.

    with torch.no_grad():

        for ims, labels, im_widths, target_lengths in testloader:

            ims, labels, im_widths, target_lengths = \
                ims.to(device), labels.to(device), im_widths.to(device), target_lengths.to(device)
            valid_pred_lengths = torch.ceil(im_widths / 4.0).type(torch.int32)  # feature map width = 1/4 image width

            out = model(ims)
            preds = nn.functional.softmax(out, dim=2)

            # decode labels
            decoded_labels = []
            i = 0
            for tlength in target_lengths:
                dl = ''.join(
                    [ctc_reverse_dictionary[int(s)] for s in labels[i:i + tlength]]
                )
                decoded_labels.append(dl)
                i += tlength

            # decode (transcribe) the model output into strings
            decoded_results = ctc_decoder.decode(preds, valid_pred_lengths)  

            # add up image number
            for dl, dr in zip(decoded_labels, decoded_results):
                pred_display = 'label: {} ===> pred: {}'.format(dl, dr)
                if dl == dr:
                    n_correct += 1
                else:
                    pred_display = '[x] ' + pred_display

                print(pred_display)
                n_ims += 1

    print('[Info] Test word accuracy = {:.1f}%'.format(100 * n_correct / n_ims))


# ==== Part 5: draw the loss and accuracy curve
def plot_loss_and_accuracies(losses, accuracies):
    '''
    :param losses: list of losses for each epoch
    :param accuracies: list of accuracies for corresponding epochs
    :return:
    '''

    # draw loss
    ax = plt.subplot(2, 1, 1)
    ax.plot(losses)
    # set labels
    ax.set_xlabel('training epoch')
    ax.set_ylabel('loss')

    # draw accuracy
    ax = plt.subplot(2, 1, 2)
    epochs, accuracies = list(zip(*accuracies))
    ax.plot(epochs, accuracies)
    # set labels
    ax.set_xlabel('training epoch')
    ax.set_ylabel('accuracy')

    # show the plots
    plt.tight_layout()
    plt.show()


# ==== Part 6: predict a new image using a trained model
def predict(model_path, im_path, norm_height, device):
    '''
    Test procedure
    ---------------
    :param model_path: path of the saved model
    :param im_path: path of an image
    '''
    # construct the model
    model = CRNN()
    model = model.to(device)
    ctc_decoder = GreedyCTCDecoder()

    # load parameters we saved in model_path
    # hint: similar to what we do in function test()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print('[Info] Load model from {}'.format(model_path))

    model.eval()  # call a function to enter evaluation mode

    # image pre-processing, similar to what we do in ListDataset()
    def resize_height_transform(image):
        height = image.size[1]
        width = image.size[0]
        scale = float(norm_height) / height
        new_height = norm_height
        new_width = int(scale * width)
        new_image = transforms.functional.resize(
            image, (new_height, new_width))
        return new_image

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(resize_height_transform),
        transforms.ToTensor(),
    ])

    im = cv2.imread(im_path)
    ax1 = plt.subplot(211)
    plt.sca(ax1)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), aspect='auto')

    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = transform(im)
    im.sub_(0.5).div_(0.5)
    im = im.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(im)
        pred = nn.functional.softmax(out, dim=2)

        # visualize the model's classification sequence
        # we can see the alignment learnt by ctc
        ax2 = plt.subplot(212)
        plt.sca(ax2)
        plt.imshow(
            pred[0, :, :].transpose(1, 0).cpu().numpy(),
            interpolation='nearest', 
            cmap=plt.cm.hot,
            vmin=0.0, vmax=1.0,
            aspect='auto')
        plt.colorbar(orientation="horizontal")

        pred_length = torch.tensor([pred.shape[1]], dtype=torch.int32)

        # decode (transcribe) the model output into strings
        decoded_result = ctc_decoder.decode(pred, pred_length)  

    plt.sca(ax1)
    plt.title('Recognition result: {}'.format(decoded_result[0]))
    print('Recognition result: {}'.format(decoded_result[0]))
    
    plt.sca(ax2)
    plt.xlabel('Timestep')
    plt.ylabel('Class')
    plt.yticks(list(range(pred.shape[2])), ['blank'] + [c for c in character_set], fontsize=6)
    plt.title("Alignment of classification sequence")
    
    plt.show()


if __name__ == '__main__':
    # set random seed for reproducibility
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # set configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train, test or predict')
    parser.add_argument('--batchsize', type=int, default=4, help='batch size')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--norm_height', type=int, default=32, help='image normalization height')

    # configurations for training
    parser.add_argument('--epoch', type=int, default=300, help='number of training epochs')
    parser.add_argument('--valInterval', type=int, default=10, help='the frequency of validation')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0., help='momentum of the SGD optimizer, only used if optim_type is sgd')
    parser.add_argument('--weight_decay', type=float, default=0., help='the factor of L2 penalty on network weights')

    # configurations for test and prediction
    parser.add_argument('--model_path', type=str, default='ctc_saved_models/model_epoch300.pth', help='path of a saved model')
    parser.add_argument('--im_path', type=str, default='data/test/word_505.png',
                        help='path of an image to be recognized')

    opt = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if opt.mode == 'train':
        train_val(
            train_im_dir=os.path.join(current_dir, 'data/train'),
            val_im_dir=os.path.join(current_dir, 'data/validation'),
            norm_height=opt.norm_height, n_epochs=opt.epoch, 
            batch_size=opt.batchsize, weight_decay=opt.weight_decay,
            lr=opt.lr, momentum=opt.momentum, valInterval=opt.valInterval, 
            device=opt.device)

    elif opt.mode == 'test':
        test(
            model_path=opt.model_path, 
            im_dir=os.path.join(current_dir, 'data/test/'), 
            norm_height=opt.norm_height, 
            batch_size=opt.batchsize, 
            device=opt.device)
            
    elif opt.mode == 'predict':
        predict(
            model_path=opt.model_path,
            im_path=opt.im_path,
            norm_height=opt.norm_height, 
            device=opt.device)

    else:
        print('mode should be train or test')
        raise NotImplementedError