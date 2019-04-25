#  Professional Masters in Big Data Program - Simon Fraser University

#  Assignment 5 (Question 5 - imagenet_finetune.py)

#  Submission Date: 29th October 2018
#  Name: Anurag Bejju
#  Student ID: 301369375
#  Professor Name: Greg Mori


import a2_p5 as a2
import torchvision.models as models
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as utils
import torch

#install Pillow==4.0.0
#install PIL
#install image

class ResNet50_CIFAR(nn.Module):
    def __init__(self):
        super(ResNet50_CIFAR, self).__init__()
        # Initialize ResNet 50 with ImageNet weights
        ResNet50 = models.resnet50(pretrained=True)
        modules = list(ResNet50.children())[:-1]
        backbone = nn.Sequential(*modules)
        # Create new layers
        self.backbone = nn.Sequential(*modules)
        self.fc1 = nn.Linear(2048, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, img):
        # Get the flattened vector from the backbone of resnet50
        out = self.backbone(img)
        # processing the vector with the added new layers
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)


def train(NUM_EPOCH,model_path):
    ## Define the training dataloader
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10('./data', download=True, train=True, transform=transform)
    #index_values = a2.get_index_values(trainset,each_class_training_size)
    #train_sampler = torch.utils.data.SubsetRandomSampler(index_values)
   # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,sampler=train_sampler, num_workers=2)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)

    ## Create model, objective function and optimizer
    if torch.cuda.is_available():
        model = ResNet50_CIFAR().cuda()
    else:
        model = ResNet50_CIFAR()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.fc1.parameters()) + list(model.fc2.parameters()),
                           lr=0.001, momentum=0.9)

    ## Do the training
    for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
    print('Finished Training')
    torch.save(model.state_dict(), model_path)
    return (model.state_dict())

def test(model_path):
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    testset = datasets.CIFAR10('./data', download=True, train=False, transform=transform)
    #index_values = a2.get_index_values(testset,each_class_test_size)
    #test_sampler = torch.utils.data.SubsetRandomSampler(index_values)
    #test_data = torch.utils.data.Subset(testset, index_values)
    #print (len(test_data))
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,num_workers=2,shuffle=True)
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model = ResNet50_CIFAR()
    if torch.cuda.is_available():
        loaded_model = torch.load(model_path)
    else:
        loaded_model = torch.load(model_path,map_location='cpu')
    model.load_state_dict(loaded_model)

    if torch.cuda.is_available():
          model = model.cuda()

    # Gets the header part of the html
    html_str = a2.html_header()
    Html_file= open("output.html","w")
    Html_file.write(html_str)

    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels = data

        # Convert_to_pil function converts the tensor image value to PIL format
        image_t = a2.convert_to_pil(inputs)

        # Image_to_base64 function converts PIL image to binary inorder to embed it into a htm without saving it onto a directory
        src_val = a2.image_to_base64(image_t)

        # Pass the inputs to the model
        if torch.cuda.is_available():
          inputs = inputs.cuda()
          labels = labels.cuda()

        outputs = model(inputs)


        # Get the predicted class for the image
        _, predicted = torch.max(outputs, 1)

        actual_class = classes[labels] #The actual class for the image
        predicted_class = classes[predicted] #The predicted class for the image
        if i % 20 == 19:    # print every 20 mini-batches
                print(str(i)+' is done')

        # This function returns the propability of the input being classified into one of the 10 classes
        if torch.cuda.is_available():
          outputs = outputs.cpu()

        prob_array = a2.get_probability(outputs)



        #This appends the image, actual class of the image, the predicted class of the image as well as the probability of the image belonging to each class.
        html_str = """
            <tr>
                <td> Sample #""" + str(i+1) + """</td>
                <td>""" + '<img src="'+src_val + """" width="60" height="60"></td>
                <td>""" + actual_class + """</td>
                <td>""" + predicted_class + """</td>"""
        for each_class_prob in prob_array[0]:
            html_str = html_str + """<td>""" + "{0:.3f}".format(round(each_class_prob,3)) + """</td>"""
        Html_file.write(html_str)

    html_str = html_str + """ </tr> </table></center></body></html>"""

    # Write it into an output.html file
    Html_file.write(html_str)
    Html_file.close()
    print('Finished Testing')


if __name__ == '__main__':

    # Enter number of Number of Epochs the training model has to run
    NUM_EPOCH = 3

    #each_class_training_size =10
    #each_class_test_size = 5
    # Enter the path for the model you want to load or the name of the model you want to save
    model_path = 'model_3epoch.pth'
    #Set this to True if you want to train a model and run test function against it or False if you want to test against a pretrained model
    train_new_model = False
    if train_new_model==True:
        #Getting the trained model
        train(NUM_EPOCH,model_path)
    #Test the model
    test(model_path)
