%%%%%%%%%%%%%%%%%%%%%%%%% Tests des hyper paramètres %%%%%%%%%%%%%%%%%%%%%
%%%%% initialisation des variables
% I = imread('house2.png');
% I = imread('lena.gif');
I = imread('clown_lumi.bmp');
[n,m] = size(I);
I2 = I;
I = double(I);
sigma = 20; % parametre du bruit
t = 5;     % 2*t+1 : taille fenetre de recherche
f = 2;      % 2*f+1 : taille du patch
h = sigma;     % parametre de filtrage
acp = 0;
maxi = 0;

%%%%% Ajout du bruit gaussien
Ibruitee = I + sigma*randn(n,m,'like',I);
Ibruitee2 = uint8(Ibruitee);

%%%%% Variation du parametre de filtrage h
liste_h = [5:5:50]';
nlm_h = zeros(n,m,size(liste_h,1));
temps_h = zeros(size(liste_h));
psnr_h = zeros(size(liste_h));
ssim_h = zeros(size(liste_h));
for i = 1:size(liste_h,1)
     tic
     nlm_h(:,:,i) = nlm_final(Ibruitee,f,t,liste_h(i),acp,maxi);
     temps_h(i) = toc;
     psnr_h(i) = psnr(I2,uint8(nlm_h(:,:,i)));
     ssim_h(i) = ssim(uint8(nlm_h(:,:,i)),I2);
end

%%%%% Variation taille du patch f
liste_f = [1:4]';
nlm_f = zeros(n,m,size(liste_f,1));
temps_f = zeros(size(liste_f));
psnr_f = zeros(size(liste_f));
ssim_f = zeros(size(liste_f));
for i = 1:size(liste_f,1)
     tic
     nlm_f(:,:,i) = nlm_final(Ibruitee,liste_f(i),t,h,acp,maxi);
     temps_f(i) = toc;
     psnr_f(i) = psnr(I2,uint8(nlm_f(:,:,i)));
     ssim_f(i) = ssim(uint8(nlm_f(:,:,i)),I2);
end

%%%%% Variation de la fenetre de recherche t
liste_t = [5:15]';
nlm_t = zeros(n,m,size(liste_t,1));
temps_t = zeros(size(liste_t));
psnr_t = zeros(size(liste_t));
ssim_t = zeros(size(liste_t));
for i = 1:size(liste_t,1)
     tic
     nlm_t(:,:,i) = nlm_final(Ibruitee,f,liste_t(i),h,acp,maxi);
     temps_t(i) = toc;
     psnr_t(i) = psnr(I2,uint8(nlm_t(:,:,i)));
     ssim_t(i) = ssim(uint8(nlm_t(:,:,i)),I2);
end

%%%%% Sauvegarde des donnees
% save('house_10max.mat','nlm_h','psnr_h','ssim_h','temps_h','liste_h',...
%     'nlm_f','psnr_f','ssim_f','temps_f','liste_f',...
%     'nlm_t','psnr_t','ssim_t','temps_t','liste_t')

% %%%%% Chargement des donnees
% load('lena_10max.mat');
% i = 1;
% load('lena_10.mat');
% i = 2;
% load('house_10.mat');
% i = 3;
% load('house_10max.mat');
% i = 4;
% 
% %%%%% Affectation des donnees
% PSNR_H(:,i) = psnr_h;
% SSIM_H(:,i) = ssim_h;
% TEMPS_H(:,i) = temps_h;
% PSNR_F(:,i) = psnr_f;
% SSIM_F(:,i) = ssim_f;
% TEMPS_F(:,i) = temps_f;
% PSNR_T(:,i) = psnr_t;
% SSIM_T(:,i) = ssim_t;
% TEMPS_T(:,i) = temps_t;
% 
% %%%%%%%%%%%%%%% A parametrer manuellement
% %%%%% affichage sigma = 10
% bar(liste_h,PSNR_H-28);xlabel('h');ylabel('psnr');title('psnr selon h / bruit = 10');legend({'lena_max','lena','house','house_max'})
% set(gca, 'YTick', [0:1:7], 'YTickLabel',[0:1:7]+28)
% 
% bar(liste_h,SSIM_H-0.6);xlabel('h');ylabel('ssim');title('ssim selon h / bruit = 10');legend({'lena_max','lena','house','house_max'})
% set(gca, 'YTick', [0:0.1:0.4], 'YTickLabel',[0:0.1:0.4]+0.6)
% 
% bar(liste_f*2+1,PSNR_F-30);xlabel('f');ylabel('psnr');title('psnr selon f / bruit = 10');legend({'lena_max','lena','house','house_max'})
% set(gca, 'YTick', [0:1:6], 'YTickLabel',[0:1:6]+30)
% 
% bar(liste_f*2+1,SSIM_F-0.8);xlabel('f');ylabel('ssim');title('ssim selon f / bruit = 10');legend({'lena_max','lena','house','house_max'})
% set(gca, 'YTick', [0:0.025:0.1], 'YTickLabel',[0:0.025:1]+0.8)
% 
% bar(liste_t*2+1,PSNR_T-32);xlabel('t');ylabel('psnr');title('psnr selon t / bruit = 10');legend({'lena_max','lena','house','house_max'})
% set(gca, 'YTick', [0:1:5], 'YTickLabel',[0:1:5]+32)
% 
% bar(liste_t*2+1,SSIM_T-0.84);xlabel('t');ylabel('ssim');title('ssim selon t / bruit = 10');legend({'lena_max','lena','house','house_max'})
% set(gca, 'YTick', [0:0.025:1], 'YTickLabel',[0:0.025:1]+0.84)
% 
% %%%%% affichage sigma = 30
% bar(liste_h,PSNR_H-18);xlabel('h');ylabel('psnr');title('psnr selon h / bruit = 10');legend({'lena','house','clown'})
% set(gca, 'YTick', [0:1:10], 'YTickLabel',[0:1:10]+18)
% 
% bar(liste_h,SSIM_H);xlabel('h');ylabel('ssim');title('ssim selon h / bruit = 10');legend({'lena','house','clown'})
% set(gca, 'YTick', [0:0.1:0.4], 'YTickLabel',[0:0.1:0.4]+0.6)
% 
% bar(liste_f*2+1,PSNR_F);xlabel('f');ylabel('psnr');title('psnr selon f / bruit = 10');legend({'lena','house','clown'})
% set(gca, 'YTick', [0:0.25:2], 'YTickLabel',[0:0.25:2]+35)
% 
% bar(liste_f*2+1,SSIM_F);xlabel('f');ylabel('ssim');title('ssim selon f / bruit = 10');legend({'lena','house','clown'})
% set(gca, 'YTick', [0:0.025:0.1], 'YTickLabel',[0:0.025:1]+0.8)
% 
% bar(liste_t*2+1,PSNR_T);xlabel('t');ylabel('psnr');title('psnr selon t / bruit = 10');legend({'lena','house','clown'})
% set(gca, 'YTick', [0:0.3:1.5], 'YTickLabel',[0:0.3:1.5]+36)
% 
% bar(liste_t*2+1,SSIM_T-0.84);xlabel('t');ylabel('ssim');title('ssim selon t / bruit = 10');legend({'lena','house','clown'})
% set(gca, 'YTick', [0:0.025:1], 'YTickLabel',[0:0.025:1]+0.84)



