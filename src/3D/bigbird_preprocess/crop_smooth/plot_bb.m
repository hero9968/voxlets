function plot_bb(bb)

p = 2;
q = 3;

subplot(p, q, 1)
imagesc(bb.grey)
axis image

subplot(p, q, 2)
imagesc(bb.rgb)
axis image

subplot(p, q, 3)
imagesc(bb.depth)
axis image

subplot(p, q, 4)
imagesc(bb.mask)
axis image

subplot(p, q, 5)
imagesc(bb.front_render)
axis image

subplot(p, q, 6)
imagesc(bb.back_render)
axis image

