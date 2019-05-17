####################
### 1,FLANN(recommonded)
####################

class FLANN_compare:

    from pyflann import * 
    from numpy import * 
    from numpy.random import * 

    def start(dataset, testset, top_numbers = 5):
        flann = FLANN() 
        ## algorithm (linear, kdtree, autotuned, means, composite)
        params = flann.build_index(dataset, algorithm="autotuned", target_precision=0.9, log_level = "info"); 
        print params 
        result, dists = flann.nn_index(testset,top_numbers, checks=params["checks"]);
        return result, dists
        
    def debug():
        dataset = rand(10000, 128) 
        testset = rand(1000, 128) 

        flann = FLANN() 
        ## algorithm (linear, kdtree, autotuned, means, composite)
        params = flann.build_index(dataset, algorithm="autotuned", target_precision=0.9, log_level = "info"); 
        print params 
        result, dists = flann.nn_index(testset,5, checks=params["checks"]);

####################
### 2,annoy
####################

class annoy_compare:

    from annoy import AnnoyIndex
    import random
    
    def start(dataset, test_vector, num_nearest=5):
    
        annoy_index = AnnoyIndex(dataset,len(dataset))
                
        neighbors = model.most_similar(test_vector, topn=num_nearest, indexer=annoy_index)
        for neighbor in neighbors:
            print(neighbor)

    def debug():

        f = 40
        t = AnnoyIndex(f)  # Length of item vector that will be indexed
        for i in xrange(1000):
            v = [random.gauss(0, 1) for z in xrange(f)]
            t.add_item(i, v)

        t.build(10) # 10 trees
        t.save('test.ann')

        # ...
        u = AnnoyIndex(f)
        u.load('test.ann') # super fast, will just mmap the file
        print(u.get_nns_by_item(0, 1000)) # will find the 1000 nearest neighbors


####################
### 3,nearpy (Numpy , Scipy, redis)
####################

class nearpy_compare:

    from nearpy import Engine
    from nearpy.hashes import RandomBinaryProjections
    
    def start(dataset, test_vector, num_nearest=5):

        # Create a random binary hash with 10 bits
        rbp = RandomBinaryProjections('rbp', 10)

        # Create engine with pipeline configuration
        engine = Engine(dataset.shape, lshashes=[rbp])

        # Index 1000000 random vectors (set their data to a unique string)
        for i,v in dataset:
            engine.store_vector(v, 'data_%d' % i)

        # Get nearest neighbours
        N = engine.neighbours(test_vector)
        
    def debug():
        # Dimension of our vector space
        dimension = 500

        # Create a random binary hash with 10 bits
        rbp = RandomBinaryProjections('rbp', 10)

        # Create engine with pipeline configuration
        engine = Engine(dimension, lshashes=[rbp])

        # Index 1000000 random vectors (set their data to a unique string)
        for index in range(100000):
            v = numpy.random.randn(dimension)
            engine.store_vector(v, 'data_%d' % index)

        # Create random query vector
        query = numpy.random.randn(dimension)

        # Get nearest neighbours
        N = engine.neighbours(query)