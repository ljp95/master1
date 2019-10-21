%%%%%%%%%%%%%%%%%%%%%% Test NLM et filtres locaux %%%%%%%%%%%%%%%%%%%%%%
%%%%%%   Initialisation des variables 
I = imread('house2.png');
% I = imread('lena.gif');
% I = imread('clown_lumi.bmp');
I2 = I;
I = double(I);
[n,m] = size(I);
sigma = 10; % parametre du bruit
t = 5;     % 2*t+1 : taille fenetre de recherche
f = 2;      % 2*f+1 : taille du patch
h = sigma;     % parametre de filtrage
acp = 9;
maxi = 1;

%%%%% Ajout du bruit 
Ibruitee = I + sigma*randn(n,m,'like',I);type_bruit = 'gaussien';
% Ibruitee = imnoise(I,'gaussien',0,0.002);type_bruit = 'gaussien';
% Ibruitee = imnoise(I2,'poisson');type_bruit = 'poisson';
% Ibruitee = imnoise(I2,'poivre et sel');type_bruit = 'poivre et sel'; % 5% de pixels
% Ibruitee = imnoise(I2,'speckle');type_bruit = 'speckle'; % I+n*I avec n uniformement distribuee de moyenne et variance 0

%%%%% affichage
Ibruitee2 = uint8(Ibruitee);
figure();imshow(I2);title('Originale');
figure();imshow(Ibruitee2);title(type_bruit);

%%%%%%%%%%%%%%%% Debruitage par differents filtres %%%%%%%%%%%%%%%%
%%%%% debruitage NLM
tic
NLM = nlm_final(Ibruitee,f,t,h,acp,maxi);
NLM2 = uint8(NLM);
time = toc
PSNR = psnr(I2,NLM2);
sSIM = ssim(I2,NLM2);

%%%%% affichage originale,bruitee,nlm,method noise
subplot(2,2,1);imshow(I2);title('Originale');
subplot(2,2,2);imshow(uint8(Ibruitee));title('Bruitee');
subplot(2,2,3);imshow(NLM2);title('NLM');
title(['NLM/Temps : ',num2str(time),'/psnr : ',num2str(PSNR),'/ssim : ',num2str(sSIM)]);
colormap gray;subplot(2,2,4);imagesc(I-NLM);title('method noise');

%%%%% affichage nlm
figure();colormap gray;imagesc(NLM2);title(['NLM/ parametre filtrage :',num2str(h),'/psnr : ',num2str(PSNR),'/ssim : ',num2str(sSIM)]);

%%%%%%%%%%%%%%%% Matlab non local means %%%%%%%%%%%%%%%%
tic
NLM = imnlmfilt(Ibruitee);
NLM2 = uint8(NLM);
figure();colormap gray;
subplot(1,2,1);imshow(NLM2);title('NLM');
subplot(1,2,2);imagesc(I-NLM);title('method noise');
PSNR = psnr(I2,NLM2)
PSNR = ssim(I2,NLM2)
time = toc

%%%%%%%%%%%%%%%% Filtre moyenneur %%%%%%%%%%%%%%%%
liste_mean = [1,2,3,4]';
figure();title('Filtre moyenneur');
for i=1:size(liste_mean,1)
    filtre = fspecial('average',2*liste_mean(i)+1);
    Imean = conv2(Ibruitee,filtre,'same');
    Imean2 = uint8(Imean);
    subplot(2,2,i);imshow(Imean2);
    title([num2str(liste_mean(i)*2+1),'/',num2str(psnr(I2,Imean2)),'/',num2str(ssim(I2,Imean2))]);
end

%%%%%%%%%%%%%%%% Filtre median %%%%%%%%%%%%%%%%
liste_median = [3,5,7,9]';
figure();colormap gray;title('Filtre médian');
for i=1:size(liste_median,1)
    Imedian = medfilt2(Ibruitee,[liste_median(i),liste_median(i)]); 
    Imedian2 = uint8(Imedian);
    subplot(2,2,i);imshow(Imedian2);
    title([num2str(liste_median(i)),'/',num2str(psnr(I2,Imedian2)),'/',num2str(ssim(I2,Imedian2))]);
end

%%%%%%%%%%%%%%%%%% Filtre gaussien %%%%%%%%%%%%%%
k = 3; l = 13;
liste_gaussien = [1:k]';
liste_sigma = [5:l]';
ssim_g = zeros(1,l,k);
psnr_g = zeros(1,l,k);
for i=1:k
    figure();colormap gray;title('Filtre gaussien');
    for j=1:l-4
        j2 = j+4;
        gauss = fspecial('gaussian',2*liste_gaussien(i)+1,liste_sigma(j));
        Igauss = conv2(Ibruitee,gauss,'same');
        Igauss2 = uint8(Igauss);
        subplot(3,3,j);imshow(Igauss2);
        ssim_g(1,j,i) = ssim(I2,Igauss2);
        psnr_g(1,j,i) = psnr(I2,Igauss2);
        title([num2str(i),'/',num2str(j2),'/',num2str(psnr_g(1,j,i)),'/',num2str(ssim_g(1,j,i))]);
    end
end


