import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F 

def str_to_int(all_char,same_start_end=True):
    stoi = {j:i+1 for i,j in enumerate(all_char)}
    if same_start_end:
        stoi['.']=0
    else:
        stoi['<s>']=0
        stoi['<e>']=27
    return stoi

def int_to_str(stoi):
    itos= {i:s for s,i in stoi.items()}
    return itos


def get_bigram_counts(words,vocabulary_size,stoi):
    count_array= torch.zeros((vocabulary_size,vocabulary_size),dtype=torch.int32) 
    for ele in words:
        
        if vocabulary_size==27:
            full = ['.'] + list(ele) + ['.'] #this is to hallucinate to start and end the word
            for a,b in zip(full,full[1:]):
                count_array[stoi[a],stoi[b]] += 1
        else:
            full = ['<s>'] + list(ele) + ['<e>'] #this is to hallucinate to start and end the word
            for a,b in zip(full,full[1:]):
                count_array[stoi[a],stoi[b]] += 1           
    return count_array



def bigram_visualization(count_array,itos,vocabulary_size):
    plt.figure(figsize=(16,16))
    plt.imshow(count_array, cmap='Blues')
    for i in range(vocabulary_size):
        for j in range(vocabulary_size):
            chstr =itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
            plt.text(j, i, count_array[i, j].item(), ha="center", va="top", color='gray')
    plt.axis('off')


def generator(seed_value):
    torch.manual_seed(seed_value)
    g = torch.Generator()
    return g


def prediction_from_distribution(probabilities, number_of_samples,generator):
    result = torch.multinomial(probabilities,num_samples=number_of_samples,replacement=True, generator=generator) # we are selecting 30 samples from the distribution
    return result

def samples_from_distribution(sample_number,count_array,itos,generator):
    for i in range(sample_number):
        out =[]
        ix =0
        while True:
            p=count_array[ix].float()
            p=p/p.sum()
            ix=prediction_from_distribution(p,1,generator).item()
            # ix= torch.multinomial(p,num_samples=1,replacement=True, generator=g).item()
            out.append(itos[ix])
            if(ix==0):
                break
        print(''.join(out))


def print_probabilities_of_each_pair(words,count_array,stoi):
    count_array = count_array.float()
    count_array = count_array/ count_array.sum(1,keepdim=True)
    for word in words[:3]:
        full = ['.'] + list(word) + ['.']
        for ch1,ch2 in zip(full,full[1:]):
            a= stoi[ch1]
            b=stoi[ch2]
            prob = count_array[a,b]
            print(f'{ch1}{ch2} : {prob:.4f}')


def different_quality_measures(words,fake_count_array,stoi):
    log_likelihood = 0.0
    n=0

    for word in words:
        full = ['.'] + list(word) + ['.']
        for ch1,ch2 in zip(full,full[1:]):
            a= stoi[ch1]
            b=stoi[ch2]
            prob = fake_count_array[a,b]
            logprob = torch.log(prob)
            log_likelihood+=logprob
            n+=1
            # print(f'{ch1}{ch2} : {prob:.4f} {logprob:.4f}')

    print(f'{log_likelihood=}')
    nll = -1*log_likelihood
    print(f'{nll=}')
    print("average of negative log likelihood = ",nll/n) #this is the average of negative log likelihood. the lower is its value , the better is the model. This summarizes the quality of the model


def input_output_for_neural_network(words,stoi):
    inp=[]
    out=[]
    for word in words:
        full_word = ['.']+list(word)+['.']
        for s,e in zip(full_word,full_word[1:]):
            inp.append(stoi[s])
            out.append(stoi[e])
    print('total number of input to neural network', len(inp))
    inp=torch.tensor(inp)
    out=torch.tensor(out)
    return inp,out




def tuning_parameters(input,output,W):
    for i in range(15):
        #forward propagation
        vector_inp = F.one_hot(input,num_classes=27).float() #this vecotrizes input  will be input to Neural network and it is better to give input to neutal network in float form
        logits= vector_inp @ W
        count = logits.exp()
        probabilities = count / count.sum(dim=-1, keepdim=True) 
        average_nll = -probabilities[torch.arange(len(input)),output].log().mean()
        loss= average_nll
        print(loss.item())

        #negative propagation
        W.grad = None  
        loss.backward()  

        #updation
        W.data += -50*W.grad


def sampling_from_neural_netwok(W,itos,generator):
   for i in range(10):
      ind = 0
      output=[]
      while True:
         vector_inp = F.one_hot(torch.tensor(ind),num_classes=27).float()
         logits= vector_inp @ W  #log counts
         counts = logits.exp()  #extracting counts from log counts
         probabilities = counts/counts.sum(0, keepdims=True)  #extracting probabilities from log counts
         ind =  prediction_from_distribution(probabilities, 1,generator).item()
         output.append(itos[ind])
         if ind==0:
            break

      print(''.join(output))

print("hello")