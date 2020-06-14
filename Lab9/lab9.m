I=double(imread('Lab1_3_5.bmp')) / 255;
figure; 
imshow(I); 
title('Исходное изображение');
PSF=fspecial('motion', 50, 25);
[J1 P1]=deconvblind(I, PSF);
figure;
imshow(J1);
title('Восстановленное изображение');