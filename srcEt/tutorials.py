


def runminimalexample(dargs):
    """
    dargs: args in dictionary form
    This code is the first minimal usage described in the README

    """

    from src import milan, milannotations

    # Load the base model trained on all available data (except ViT):
    decoder = milan.pretrained('base') ; print('decoder loaded...')

    # Load some neurons to describe; we'll use unit 10 in layer 9.
    """
    # dataset = milannotations.load('dino_vits8/imagenet') ; print('dataset loaded...')
    
    ^^^ This line of code above suggested in the README is NOT friendly to small machine. 
    It will terminate  most likely because of the lack of memory.

    The following is used to replicate the above, except ONLY the relevant layer is loaded
    to save time
    """
    dataset = milannotations.load('dino_vits8/imagenet',**dargs) ; print('dataset loaded...')    
    sample = dataset.lookup(dargs['lookup_layer'], dargs['lookup_unit']) # 

    # Caption the top images.
    #  torch.Size([1, 15, 3, 224, 224]) torch.Size([1, 15, 1, 224, 224])
    print(sample.images[None].shape, sample.masks[None].shape)
    plot_sample(sample)

    outputs = decoder(sample.images[None], masks=sample.masks[None])
    # print(outputs.captions[0])
    print(outputs.captions)


def plot_sample(sample):
    imgs = sample.images[None]
    _,n,c,h,w = imgs.shape

    def tonumpy(img):
        return img.clone().detach().cpu().numpy().transpose(1,2,0)

    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(n):
        plt.gcf().add_subplot(3,5,i+1)
        plt.gca().imshow( tonumpy(imgs[0,i]))
    plt.show()