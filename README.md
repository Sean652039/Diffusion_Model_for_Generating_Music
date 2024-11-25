# Stable diffusion Music

The training plot size is 768\*512, which can be seen as 64 subplots with size 96\*64 combined together. 96 represents pitch range from 17 -112(The original range from the datasets is 21 - 108, the reason for 96 is simply as 96 is divisible by 768). 64 represents 0.64s, because the time step is 0.01s.

So in that case, we can just reorganise the roll plot from 96\*4096 into 768(8\*96)\*512(8\*64).



![piano_roll_0](https://cdn.jsdelivr.net/gh/Sean652039/pic_bed@main/uPic/piano_roll_0.jpg)