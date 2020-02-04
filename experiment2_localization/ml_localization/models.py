import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

# collect all models in a dictionary for easy configuration
models = dict()

class MLP(chainer.Chain):
    '''
    Multi-Layer Perceptron with variable number of layers
    and hidden units.

    Parameters
    ----------
    layers: list of int
        The number of hidden units per layer
    n_out: int
        The number of outputs of the network
    '''

    def __init__(self, layers, n_out):
        super(MLP, self).__init__()
        with self.init_scope():

            layers = [None] + layers + [n_out]

            # All the hidden layers
            self.weights = chainer.ChainList(
                    *[ L.Linear(l_in, l_out) for l_in, l_out in zip(layers[:-1],layers[1:]) ]
                    )

    def __call__(self, x):
        h = x
        for W in self.weights[:-1]:
            h = W(h)
            h = F.relu(h)
        return self.weights[-1](h)

models['MLP'] = MLP


class ResBlock(chainer.Chain):
    '''
    A two layers residual block

    Parameters
    ----------
    n: int
        the width of the block
    '''

    def __init__(self, n, n_hidden):
        super(ResBlock, self).__init__()
        with self.init_scope():

            self.hidden1 = L.Linear(n, n_hidden)
            self.hidden2 = L.Linear(n_hidden, n)

    def __call__(self, x):
        h = x
        h = F.relu(self.hidden1(h))
        h = F.relu(self.hidden2(h))

        return x + h

models['ResBlock'] = ResBlock


class ResReg(chainer.Chain):

    def __init__(self, n_res, n_res_in, n_hidden, n_out, dropout=None):
        super(ResReg, self).__init__()
        with self.init_scope():

            self.dropout = dropout
            self.input = L.Linear(None, n_res_in)
            self.res_blocks = chainer.ChainList(
                    *[ResBlock(n_res_in, n_hidden) for n in range(n_res)]
                    )
            self.output = L.Linear(n_res_in, n_out)

    def __call__(self, x):

        h = F.relu(self.input(x))

        if self.dropout is not None:
            h = F.dropout(h, ratio=self.dropout)

        for R in self.res_blocks:
            h = R(h)

        return self.output(h)

models['ResReg'] = ResReg


class BlinkNet(chainer.Chain):
    '''
    Parameters
    ----------
    locations: ndarray (n_sensors, n_dim)
        The locations of the sensors
    net_name: str
        The name of the network model to use
    *net_args:
        The positional arguments of the network
    **net_kwargs: 
        The keyword arguments of the network
    '''

    def __init__(self, locations, net_name, *net_args, **net_kwargs):
        super(BlinkNet, self).__init__()
        with self.init_scope():

            self.locations = chainer.Parameter(np.array(locations, dtype=np.float32)[None,:,:])
            self.network = models[net_name](*net_args, **net_kwargs)
            self.eye = chainer.Parameter(np.eye(self.locations.shape[1], dtype=np.float32))
            self.linear_x = L.Linear(None, self.locations.shape[1])
            self.linear_y = L.Linear(None, self.locations.shape[1])

    def __call__(self, x):

        '''
        if x.ndim == 3:
            x = np.squeeze(x)

        x_loc_rep = F.repeat(self.locations[:,:,0], x.shape[0], axis=0)
        y_loc_rep = F.repeat(self.locations[:,:,1], x.shape[0], axis=0)

        max_loc = np.argmax(x, axis=1)
        loc = self.eye[max_loc,:]
        x = F.concat((x, x_loc_rep, y_loc_rep), axis=1)
        '''

        h = self.network(x)
        h_x = self.linear_x(F.relu(h))
        h_y = self.linear_y(F.relu(h))
        h = F.concat((h_x[:,:,None], h_y[:,:,None]), axis=2)

        loc_bc = F.broadcast_to(self.locations, h.shape)

        return F.sum(h * loc_bc, axis=1)

models['BlinkNet'] = BlinkNet


class SimplerBlinkNet(chainer.Chain):
    '''
    Parameters
    ----------
    locations: ndarray (n_sensors, n_dim)
        The locations of the sensors
    net_name: str
        The name of the network model to use
    *net_args:
        The positional arguments of the network
    **net_kwargs: 
        The keyword arguments of the network
    '''

    def __init__(self, locations, n_res_blk, n_res_hidden, dropout=None):
        super(SimplerBlinkNet, self).__init__()
        with self.init_scope():

            n_blinkies = len(locations)

            # parameters
            self.locations = chainer.Parameter(np.array(locations, dtype=np.float32))
            self.eye = chainer.Parameter(np.eye(n_blinkies, dtype=np.float32))
            self.dropout = dropout

            # sub_components
            self.input = L.Linear(None, self.locations.shape[0])
            self.output = L.Linear(self.locations.shape[0], self.locations.shape[0])
            n_res_in = self.locations.shape[0]
            self.res_blocks = chainer.ChainList(
                    *[ResBlock(n_blinkies, n_res_hidden) for n in range(n_res_blk)]
                    )

    def __call__(self, x):

        h = F.relu(self.input(x))

        if self.dropout is not None:
            h = F.dropout(h, ratio=self.dropout)

        for R in self.res_blocks:
            h = R(h)

        h = F.relu(self.output(h))

        return F.matmul(h, self.locations)

models['SimplerBlinkNet'] = SimplerBlinkNet


class MaxLocNet(chainer.Chain):
    '''
    Parameters
    ----------
    locations: ndarray (n_sensors, n_dim)
        The locations of the sensors
    net_name: str
        The name of the network model to use
    *net_args:
        The positional arguments of the network
    **net_kwargs: 
        The keyword arguments of the network
    '''

    def __init__(self, locations, k_max, net_name, *net_args, **net_kwargs):
        super(MaxLocNet, self).__init__()
        with self.init_scope():

            self.locations = chainer.Parameter(np.array(locations, dtype=np.float32))
            self.k_max = k_max
            self.network = models[net_name](*net_args, **net_kwargs)
            self.eye = chainer.Parameter(np.eye(self.locations.shape[0], dtype=np.float32))

    def __call__(self, x):

        n_batch, n_blinkies = x.shape

        # find blinky most likely closest to source
        max_loc = np.argsort(x, axis=1)[:,-self.k_max:]
        loc = []
        for i in range(self.k_max):
            loc.append(self.locations[max_loc[:,i],:])
        loc = F.concat(loc, axis=1)

        h = F.concat((x, loc), axis=1)
        h = self.network(h)

        loc = F.reshape(loc, (-1, self.k_max, self.locations.shape[1]))

        return np.sum(loc * F.broadcast_to(h[:,:,None], loc.shape), axis=1)

models['MaxLocNet'] = MaxLocNet
