import sys
import numpy as np
import matplotlib.pyplot as plt
actions=['Cook.Cleandishes','Cook.Cleanup','Cook.Cut', 'Cook.Stir', 'Cook.Usestove', 
'Cutbread', 'Drink.Frombottle', 'Drink.Fromcan', 'Drink.Fromcup', 'Drink.Fromglass',
'Eat.Attable', 'Eat.Snack', 'Enter','Getup', 'Laydown', 'Leave', 
'Makecoffee.Pourgrains', 'Makecoffee.Pourwater', 'Maketea.Boilwater', 
'Maketea.Insertteabag', 'Pour.Frombottle', 
'Pour.Fromcan', 'Pour.Fromkettle', 'Readbook', 'Sitdown', 'Takepills', 
'Uselaptop', 'Usetelephone', 'Usetablet', 'Walk', 'WatchTV']


filename=sys.argv[1]
num_cls=int(sys.argv[2])

con_matrix=np.zeros((num_cls,num_cls))
accuracy_matrix=np.zeros((num_cls,num_cls))

with open(filename,'r') as result:
    results=result.readlines()
for i in results:
    predict=int(i.split(',')[0])
    gt=int(i.split(',')[1])
    con_matrix[gt, predict]+=1

sum_gt=con_matrix.sum(axis=1)
sum_t=sum_gt.sum(axis=0)
tp=0
acc_tot=0

for j, row in enumerate(con_matrix):
    accuracy_matrix[j]=row/sum_gt[j]
    #print('samples for '+actions[j]+': ', sum_gt[j])
for a in range(num_cls):
    tp+=con_matrix[a,a]
    print('accuracy_'+actions[a]+' : ', accuracy_matrix[a,a])
    #print('accuracy : ', accuracy_matrix[a,a])
    acc_tot+=accuracy_matrix[a,a]
print('\ntotal accuracy: ', tp/sum_t )
print('\naverage accuracy: ', acc_tot/(num_cls))

# plot
fig=plt.figure()
ax=fig.add_subplot(111)
#ax.set_xticks(range(1,62,2))
#ax.set_yticks(actions)
ax.set_xlabel("Predict actions")
ax.set_ylabel("GT actions")

ax.imshow(accuracy_matrix)
plt.savefig('confusion_matrix.png')
print('Figure saved as confusion_matrix.png')
