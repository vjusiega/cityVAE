# cityVAE

CityVAE is a conceptual machine learning model that takes in 2D height maps of neighborhoods and generates new maps within the design space of the existing maps. These interpolations can be hard to navigate due to the sheer number of them and so my partner, Jackie Lin, and I used a variational autoencoder (VAE) to create five abstract variables that can be used as sliders to explore all of the possibilities. 

We imagined that this tool would be of interest to architects and city planners who may want to draw inspiration from an existing urban landscape to retrofit new buildings or create entirely new districts that still follow the rhythm of the city. 

The actual machine learning model in this repo gives some mixed results. New testing data is recreated with 95% accuracy using the corresponding encoded variables but the interpolations / neighborhood models between these concrete points leave much to be desired. We believe that much of the error came from the disassociation of the grid blocks in a given height map, so an area that should have been a street or a building was not recognized as one entity in the model. Different input data, perhaps with a better anchor for street values (such as -1000 elevation instead of just 0) or a different machine learning model (such as a GAN) might have solved our issues. The difficulty with a VAE was having all inputs be of the exact same type and length. 

This was a final project for 4.s42 - Machine Learning for Design at MIT. 
