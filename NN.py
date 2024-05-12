import random
import numpy as np
import matplotlib.pyplot as plt

class neuralnet(object):

    def __init__(self, layers):
        self.layers = layers
        self.b = [np.random.randn(y) for y in layers[1:]]
        self.w = [np.random.randn(y, x) for x,y in zip(layers[:-1], layers[1:])]

    def printBiases(self):
        index = 0
        for row in self.b:
            print(f'\nLayer {index+1}')
            print('\n'.join([''.join(['{:+.2E}  '.format(float(item)) for item in row.tolist()])]))
            index +=1

    def printWeights(self):
        index = 0
        for layer in self.w:
            print(f'\nLayer {index} -> {index+1}')
            print('\n'.join([''.join(['{:+.2E}  '.format(float(item)) for item in row])  for row in layer.tolist()]))
            index += 1

    def output(self, input_layer):
        phi = input_layer
        layer = 0
        for b,w in zip(self.b, self.w):
            phi = self.sigmoid(np.dot(w, phi) + b)
        return phi

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))


    def draw(self, sizeX = 12, sizeY = 12, left = 0.1, right =0.9, bottom = 0.1, top = 0.9, showWeights = True, showBias = True, 
             color_c = 'r', color_l = 'b', color_t = 'k', lineWidth = 0.5):
        fig = plt.figure(figsize=(sizeX, sizeY))
        ax = fig.gca()
        ax.set_xlim([-0.0, 1.0])
        ax.set_ylim([-0.0, 1.0])
        ax.axis('off')
        ax.set_aspect('equal')
        layer_sizes = self.layers
        coefs_ = self.w
        intercepts_ = self.b
        
        n_layers = len(layer_sizes)
        v_spacing = (top - bottom)/float(max(layer_sizes))
        h_spacing = (right - left)/float(len(layer_sizes) - 1)
        v_center = (top + bottom)/2.
        
        # Input-Arrows
        layer_top_0 = v_spacing*(layer_sizes[0] - 1)/2. + v_center
        for m in range(layer_sizes[0]):
            plt.arrow(left-0.18, layer_top_0 - m*v_spacing, 0.12, 0,  lw =1, head_width=0.01, head_length=0.02)
        
        # Nodes
        for n, layer_size in enumerate(layer_sizes):
            layer_top = v_spacing*(layer_size - 1)/2. + v_center
            for m in range(layer_size):
                circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/8., color='w', ec=color_c, zorder=4)
                if n == 0:
                    plt.text(left-0.14, layer_top - m*v_spacing - 0.01, r'$X_{'+str(m+1)+'}$', fontsize=15, color = color_t)
                elif (n_layers == 3) & (n == 1):
                    plt.text(n*h_spacing + left+0.00, layer_top - m*v_spacing+ (v_spacing/8.+0.01*v_spacing), r'$H_{'+str(m+1)+'}$', fontsize=15, color = color_t )
                elif n == n_layers -1:
                    plt.text(n*h_spacing + left+0.10, layer_top - m*v_spacing - 0.01, r'$Y_{'+str(m+1)+'}$', fontsize=15, color = color_t)
                ax.add_artist(circle)
        # Bias-Nodes
        if(showBias):
            for n, layer_size in enumerate(layer_sizes):
                if n < n_layers -1:
                    x_bias = (n+0.5)*h_spacing + left
                    y_bias = top + 0.005
                    circle = plt.Circle((x_bias, y_bias), v_spacing/8., color='w', ec=color_c, zorder=4)
                    plt.text(x_bias-(v_spacing/8.+0.10*v_spacing+0.01), y_bias, r'$1$', fontsize=15, color = color_t)
                    ax.add_artist(circle)   
        # Edges
        # Edges between nodes
        for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + v_center
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + v_center
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left], [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c=color_l, linewidth=lineWidth)
                    ax.add_artist(line)
                    xm = (n*h_spacing + left)
                    xo = ((n + 1)*h_spacing + left)
                    ym = (layer_top_a - m*v_spacing)
                    yo = (layer_top_b - o*v_spacing)
                    rot_mo_rad = np.arctan((yo-ym)/(xo-xm))
                    rot_mo_deg = rot_mo_rad*180./np.pi
                    xm1 = xm + (v_spacing/8.+0.05)*np.cos(rot_mo_rad)
                    if n == 0:
                        if yo > ym:
                            ym1 = ym + (v_spacing/8.+0.08)*np.sin(rot_mo_rad)
                        else:
                            ym1 = ym + (v_spacing/8.+0.08)*np.sin(rot_mo_rad)
                    else:
                        if yo > ym:
                            ym1 = ym + (v_spacing/8.+0.08)*np.sin(rot_mo_rad)
                        else:
                            ym1 = ym + (v_spacing/8.+0.08)*np.sin(rot_mo_rad)
                    if(showWeights):
                        plt.text( xm1, ym1, str(round(coefs_[n][o, m],4)), rotation = rot_mo_deg,  fontsize = 10, color = color_t)
        # Edges between bias and nodes
        if(showBias):
            for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if n < n_layers-1:
                    layer_top_a = v_spacing*(layer_size_a - 1)/2. + v_center
                    layer_top_b = v_spacing*(layer_size_b - 1)/2. + v_center
                x_bias = (n+0.5)*h_spacing + left
                y_bias = top + 0.005 
                for o in range(layer_size_b):
                    line = plt.Line2D([x_bias, (n + 1)*h_spacing + left], [y_bias, layer_top_b - o*v_spacing], c=color_l, linewidth=lineWidth)
                    ax.add_artist(line)
                    xo = ((n + 1)*h_spacing + left)
                    yo = (layer_top_b - o*v_spacing)
                    rot_bo_rad = np.arctan((yo-y_bias)/(xo-x_bias))
                    rot_bo_deg = rot_bo_rad*180./np.pi
                    xo2 = xo - (v_spacing/8.+0.01)*np.cos(rot_bo_rad)
                    yo2 = yo - (v_spacing/8.+0.01)*np.sin(rot_bo_rad)
                    xo1 = xo2 -0.11 *np.cos(rot_bo_rad)
                    yo1 = yo2 -0.08 *np.sin(rot_bo_rad)
                    plt.text( xo1, yo1,str(round(intercepts_[n][o],4)), rotation = rot_bo_deg, fontsize = 10, color = color_t)    
                    
        # Output-Arrows
        layer_top_0 = v_spacing*(layer_sizes[-1] - 1)/2. + v_center
        for m in range(layer_sizes[-1]):
            plt.arrow(right+0.015, layer_top_0 - m*v_spacing, 0.16*h_spacing, 0,  lw =1, head_width=0.01, head_length=0.02)
        
        return fig
        
        
            
         
        