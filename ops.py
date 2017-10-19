import theano
import torch 
import numpy as np

class bp(theano.Op):
    '''
        Theano.Op for backward pass used for `fp` op.
        Do not use it explicitly in your graphs. 
    '''
    def __init__(self, net, debug, dtype):
        self.net= net
        
        self.output_   = None
        self.input_    = None
        self.input_np_ = None
        
        self.debug = debug
        self.dtype = dtype
        
    __props__ = ()
 
    def make_node(self, x, y):
        x_ = theano.tensor.as_tensor_variable(x)
        y_ = theano.tensor.as_tensor_variable(y)
        return theano.gof.Apply(self, [x_, y_], [x_.type()])
 
    def perform(self, node, inputs, output_storage): 
        '''
            Actual backward pass computations. 
            We will do some kind of caching:
                Check if the input is the same as the stored one during forward pass 
                If it is the same -- do only backward pass, if it is different do forward pass again here
        '''
        
        input = inputs[0]
        grad_output = inputs[1]

        if self.debug: print('Backward pass:')

        # Caching
        if self.input_np_ is not None:
            if np.all(np.allclose(inputs[0], self.input_np_)):
                # assume np.all(np.allclose(output_var.data.cpu().numpy(), self.output_.data.cpu().numpy()))
                
                output_var = self.output_
                input_var = self.input_
                
                if self.debug: print('\t1)Forward in backward: cached')
            else:
                assert False, 'Buffer does not match input, IT\'s A BUG'
        else:
            
            input_var = torch.autograd.Variable(torch.from_numpy(input).type(self.dtype), requires_grad=True)
            output_var = self.net(input_var)
            
            if self.debug: print('\t1)Forward in backward: compute')
                
        
        if self.debug: print('\t2) Backward in backward')
        
        # Backward
        grad = torch.from_numpy(grad_output).type(self.dtype)
        output_var.backward(gradient = grad)
            
        # Put result in the right place
        output_storage[0][0] = input_var.grad.data.cpu().numpy().astype(inputs[0].dtype)

    def grad(self, inputs, output_grads):
        assert False, 'We should never get here'
        return [output_grads[0]]
 
    def __str__(self):
        return 'backward_pass'

class pytorch_wrapper(theano.Op):
    '''
        This is a theano.Op that can evaluate network from pytorch
        And get its gradient w.r.t. input 
    '''
    def __init__(self, net, debug=False, dtype=torch.FloatTensor):
        self.net = net.type(dtype)
        self.dtype = dtype
        
        self.bpop = bp(self.net, debug, dtype) 
        self.debug = debug
    __props__ = ()
 
    def make_node(self, x):
        x_ = theano.tensor.as_tensor_variable(x)
        return theano.gof.Apply(self, [x_], [x_.type()])
 
    def perform(self, node, inputs, output_storage):
        '''
            In this function we should compute output tensor
            Inputs are numpy array, so it's easy
        '''
        if self.debug: print('Forward pass')
        
        # Wrap input into variable
        input = torch.autograd.Variable(torch.from_numpy(inputs[0]).type(self.dtype), requires_grad=True)
        out = self.net(input)
        out_np = out.data.cpu().numpy().astype(inputs[0].dtype)
        
        # Put output to the right place 
        output_storage[0][0] = out_np
        
        
        self.bpop.output_ = out
        self.bpop.input_ = input
        self.bpop.input_np_ = inputs[0]
     
    def grad(self, inputs, output_grads):
        '''
            And `grad` should operate TheanoOps only, not numpy arrays
            So the only workaround I've found is to define another TheanoOp for backward pass and call it
        '''
        return [self.bpop(inputs[0], output_grads[0])]

    def __str__(self):
        return 'forward_pass'