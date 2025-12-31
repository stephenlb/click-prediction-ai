# click prediction

this code is originally something steven did on stream and I had time so i played with it a lot.
the graphics is not by me but the prediction is

I am gona do my best to explain the thought process for what is happening.
note that this is vibe coded 😅 (gpt5.2 chat client) i really did not mean for this to get as much attention as it did 


# first principles
so after playing around with 1hot encoding and cnns/transformers it became clear that we simply do not have the data to train a big model.
so fundementally we are basically trying the smallest network we can get away with.
we also ideally want to not require too many examples before we can start training.

this naturally means that we want conv nets and linear as they work great with small amounts of parameters.

also note that this is continious training so we need to keep the network elastic.

# elasticity
so after playing around with a lot of models I noticed that they can sometimes get stuck on a paticular pattern.
sigmoid activation has a nasty tendency to have vanishing derivative on the edges and thus we are using no activation.
which means yes we can predict clicks off screen but its worth it

another issue is if at any points 2 neurons have the exact same weights they would stay the same for ever.
this gradually leads to them being locked forever and our model effectively having 1 less neuron.
ALSO if something goes very negative into the ReLU we get those dead neurons which stay 0 and have 0 derivative.

to try and solve both we give weights some random movment by giving small noise to the gradients.
we also do dropout to encourage neurons to be diffrent.

# skip connections
note that a lot of times a click is directly related to the input so adding that to the last prediction head does wonders.

in general because we want training to adapt quickly we want the least amount of distance between each weight and the cost.
the network is agressively optimized to be short in terms of layers.

originally i had a bunch of skip connections but it seems to work the same way either way.
some refactor got rid of parts of it but I think its actually nicer this way.

# training 
so we are using a weighted loss function that exponentially decays in time.
the first part of training uses all of it and lower decay.
while the last part uses only a few examples and a more agressive decay.

# activations
tried a lot relu is best (softmin low key great)

