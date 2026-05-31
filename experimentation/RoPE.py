# first we will look at the original embeddings how they work
# Visualising it is important for understanding RoPE
# Fixed Wave Signatures

import torch
import matplotlib.pyplot as plt

# even numbered column i
def sinewave(pos, i, d):
    return torch.sin( pos / (torch.pow(10000, ((2 * i) / d))))

# odd numbered column i
def coswave(pos, i, d):
    return torch.cos( pos / (torch.pow(10000, ((2 * i) / d))))

def RoPE_2d(sinx,cosx, x1, x2):
    x_axis = x1 * cosx - x2 * sinx
    y_axis = x1 * sinx + x2 * cosx
    return x_axis, y_axis

def plotwavesign(x_axis, y_axis):
    plt.figure(figsize=(6, 6))

    # Plot a static point at the origin (0,0)
    plt.plot(0, 0, marker='o', color='black')
    plt.text(-0.3, -0.3, '(0,0)', fontsize=10)

    # 2. Loop through arrays to plot lines and center labels
    for i, (x, y) in enumerate(zip(x_axis, y_axis)):
        point_name = f'p{i+1}'
        
        # Plot line from (0,0) to (x,y)
        plt.plot([0, x], [0, y], marker='o', linestyle='--')
        
        # Calculate the midpoint of the line
        mid_x = x / 2
        mid_y = y / 2
        
        # Add text label directly on the line
        # bbox creates a clean background box so the line doesn't cut through the text
        plt.text(mid_x, mid_y, point_name, fontsize=10, weight='bold',
                horizontalalignment='center', verticalalignment='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))

    # 3. Add layout configurations
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Labels Placed on the Lines')
    plt.grid(True)

    # Display the final plot
    plt.show()


if __name__ == "__main__" :

    x = torch.tensor([[0.25, -0.47], [0.25, -0.47]])
    d = torch.tensor(x.shape[-1])
    x_axis = []
    y_axis = []
    # positional embeddings value using fixed wave signatures
    for pos in range(1,10) :
        # print(f"pos = {pos}, [{sinewave(pos, 0, d), coswave(pos, 1, d)}]")
        x_axis.append((pos*0.1 + sinewave(pos, 0, d)))
        y_axis.append((pos*0.1 + coswave(pos, 1, d)))
    # print("x values:",x_axis)
    # print("y values", y_axis)

    # as we can see their is no pattern in the graph which we are generating using the sinusoidal waves
    # the actual values of the token embedding is getting affected since we are adding the positional encoding to the token encoding
    # the perplexity of the model skyrockets or explodes beyong the training length (sequence length)
    # this perplexity explosion happens for both sinusoidal and absolute positional embeddings
    # So We use RoPE to fix this issue
    # plotwavesign(x_axis, y_axis)

    # The logic behind Rope is we break down the dimension of each position in pair of 2 so we have d/2 pairs
    # We rotate these pairs counterclockwise using a polar cordinates [[cos(x),-sin(x)],[sin(x),cos(x)]] 
    # each pair is rotated by different angles so they rotate at different speeds x = pos * 10000 ^ (-2*(i-1)/d) where i= (1,2,3,...,d/2)
    # after this change if we have to calculate the inner dot product of the query(m) and key(n) it only depends on the relative position offset which is (n-m)
    input_x = []
    for pos in range (1,10):
        input_x.append([0.1 * pos, 0.1 * pos])
    input_x = torch.tensor(input_x)

    print(input_x)

    rope_x = []
    rope_y = []

    for pos in range(1,7):
        sinx = sinewave(pos, 0, d)
        cosx = coswave(pos, 0, d)
        a, b = RoPE_2d(sinx, cosx, 0.2 * pos, 0.2 * pos)
        rope_x.append(a)
        rope_y.append(b)
    
    # this gives a proper pattern to the graph instead of random values
    plotwavesign(rope_x, rope_y)
