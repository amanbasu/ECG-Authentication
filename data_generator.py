import glob
import numpy as np
from PIL import Image

class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self, path, data_subsets=["train", "val"]):
        
        # gets all file paths (separate for each person)
        self.file_set = [file for file in (glob.glob(path + '/ecg-id-database-filter/Person_{:02d}/rec_*.png'.format(num)) for num in range(1, 91))]
        
        # defining train and val set
        self.val_set = [1, 6, 13, 15, 21, 25, 29, 38, 41, 44, 52, 62, 68, 74, 80, 89]
        self.train_set = list(range(1, 91))
        for ele in sorted(self.val_set, reverse=True):  
            del self.train_set[ele-1]

    def get_batch(self, batch_size, s="train"):
        """Create batch of n pairs, half same class, half different class"""

        if s=='train':
          ids = self.train_set
        else:
          ids = self.val_set

        w, h = 144, 224

        #initialize 2 empty arrays for the input image batch
        pairs = [np.zeros((batch_size, w, h, 1)) for i in range(2)]
        #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets = np.zeros((batch_size,))
        targets[batch_size//2:] = 1

        for i in range(batch_size):
            # select random person
            idx_1, idx_2 = np.random.choice(ids, 2, replace=False).tolist()
            # select random ecg sample of the person
            pair_x, pair_y = np.random.choice(self.file_set[idx_1-1], 2, replace=False).tolist()

            # load and format the image
            img = Image.open(pair_x)
            img = img.resize((h, w))
            pairs[0][i,:,:,:] = np.array(img)[:,:,0:1] / 255

            #pick images of same class for 1st half, different for 2nd
            if i < batch_size // 2:
                pair_y = np.random.choice(self.file_set[idx_2-1], 1, replace=False)[0]

            img = Image.open(pair_y)
            img = img.resize((h, w))
            pairs[1][i,:,:,:] = np.array(img)[:,:,0:1] / 255

        return pairs, targets
    
    def generate(self, batch_size, s="train"):
        """a generator for batches, so model.fit_generator can be used. """
        while True:
            pairs, targets = self.get_batch(batch_size,s)
            yield (pairs, targets)    

    def make_oneshot_task(self, N, s="val"):
        """Create pairs of test image, support set for testing N way one-shot learning. """

        if s=='train':
          ids = self.train_set
        else:
          ids = self.val_set

        w, h = 144, 224

        # select random people
        idx = np.random.choice(ids, N, replace=False)
        true_idx = idx[0]      # test person
        false_idx = idx[1:]    # random people

        pairs = [np.zeros((N, w, h, 1)) for i in range(2)]

        # contains N-1 sample of random people, 1 sample of test person
        support_set = np.zeros((N, w, h, 1))
        # contains N sample of test person (1 person)
        test_image = np.zeros((N, w, h, 1))

        targets = np.zeros((N,))
        targets[0] = 1

        # gets file path of N+1 sample of test person
        pair_x = np.random.choice(self.file_set[true_idx-1], N+1, replace=False).tolist()
        support_set[0] = np.array(Image.open(pair_x[0]).resize((h, w)))[:,:,0:1] / 255

        j = 0
        for i in pair_x[1:]:
          test_image[j] = np.array(Image.open(i).resize((h, w)))[:,:,0:1] / 255
          j += 1

        for i in range(1, N):
          pair_y = np.random.choice(self.file_set[false_idx[i-1]], 1, replace=False)[0]
          support_set[i] = np.array(Image.open(pair_y).resize((h, w)))[:,:,0:1] / 255
            
        # shuffle data
        targets, test_image, support_set = shuffle(targets, test_image, support_set)
        pairs = [test_image, support_set]

        return pairs, targets
    
    def test_oneshot(self, model, N, k, s="val", verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        n_correct = 0
        if verbose:
            print("\nEvaluating model on {} random {} way one-shot learning tasks ...".format(k,N))
        
        acc = []
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N, s)
            probs = model.predict(inputs).reshape((N,))

            # check accuracy
            acc.append(np.argmax(probs)==np.argmax(targets))
            
        percent_correct = 100*sum(acc)/len(acc)

        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct
    
    def train(self, model, epochs, verbosity):
        # train the model
        model.fit_generator(self.generate(batch_size))
