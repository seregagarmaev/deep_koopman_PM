

loss_f=nn.L1Loss()
from despawn import OCTOPUS as BIG

Net = BIG.Octopus(np.shape(DataT)[1],
                  Input_Level=9,
                  Input_Archi="DWT",#"DWT" for Despawn, "WPT" for frequency band of equal size (then Output_size = 2**Input_Level)
                  Filt_Tfree=False)
data_inputs = GG.Generate(batch_size)#Generator corresponding to your dataset (output here Batch_size x Times)
data_inputs=data_inputs.unsqueeze(1)
preds = Net.T(data_inputs)
preds = Net.iT(preds)
plt.plot(preds[0].detach().squeeze().numpy())
plt.plot(data_inputs[0].detach().squeeze().numpy())


len_S=np.shape(DataT)[1]
optimizer=torch.optim.Adam(Net.parameters(), lr=0.0001)

num_epochs = 300
batch_size = 8

loss_list     = []
accuracy_list = []

lambda=1

for epoch in range(num_epochs):

      # Training Loop
    Net.train()
    runing_loss, num_step = 0., 0.
    for i in range(10):
        
        ## Step 1: Move input data to the same device as the model
        data_inputs = GG.Generate(batch_size)
        data_inputs = data_inputs.unsqueeze(1)
        
        ## Step 2: Forwad pass
        preds = Net.T(data_inputs)
        emb=torch.abs(torch.cat(preds,axis=2)).squeeze(1)
        
        ## Step 3: Calculate the loss
        loss = loss_f(Net.iT(preds), data_inputs)+lambda*loss_f(emb,torch.zeros_like(emb))
  
  
        ## Store the loss for visualisation
        runing_loss += loss.item()
        num_step += 1
  
        ## Step 4:
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

    Net.eval()
    data_test, true_v = GG.Test()
    data_test=data_test.unsqueeze(1)
    pred = Net.iT(Net.T(data_test))
    error=np.sum(np.square(data_test.detach().squeeze().numpy()-pred.detach().squeeze().numpy()),axis=1)
    score = metrics.roc_auc_score(true_v, error)
    print(score)
    loss_list.append(runing_loss/num_step)
    print(runing_loss/num_step)



Net.eval()
data_test, true_v = GG.Test_D()
pred = Net.iT(Net.T(data_test))
a=np.sum(np.square(data_test[:,:,len_S:2*len_S].detach().squeeze().numpy()-pred[:,:,len_S:2*len_S].detach().squeeze().numpy()),axis=1)
b=Net.Embed(Net.T(data_test)).detach().numpy()

error_all=np.concatenate((a[:,np.newaxis],b),axis=1)
for i in range(len(error_all.T)):
    if i==0:
        error_all[:,i]=10*(error_all[:,i]-np.mean(error_all[:,i]))/(np.std(error_all[:,i]-np.mean(error_all[:,i])))        
    else:
        error_all[:,i]=(error_all[:,i]-np.mean(error_all[:,i]))/(np.std(error_all[:,i]-np.mean(error_all[:,i])))
error=np.sum(error_all,axis=1)

from sklearn import metrics
score = metrics.roc_auc_score(true_v, error)
print(score)
plt.plot(error)

error_all=np.concatenate((a[:,np.newaxis],b),axis=1)
for i in range(len(error_all.T)):
    if i==0:
        error_all[:,i]=(error_all[:,i]-np.mean(error_all[:,i]))/(np.std(error_all[:,i]-np.mean(error_all[:,i])))        
    else:
        error_all[:,i]=(error_all[:,i]-np.mean(error_all[:,i]))/(np.std(error_all[:,i]-np.mean(error_all[:,i])))
error=np.sum(error_all,axis=1)

from sklearn import metrics
score = metrics.roc_auc_score(true_v, error)
plt.figure(2)
print(score)
plt.plot(error)