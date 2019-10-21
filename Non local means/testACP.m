%%%%%%%%%%%%%%%%%%%%%%% Tests ACP  %%%%%%%%%%%%%%%%%%%%%%%%
I = imread('lena.gif');
I2 = I;
I = double(I);
[n,m] = size(I);
sigma = 10; % parametre du bruit
t = 5;     % 2*t+1 : taille fenetre de recherche
f = 4;      % 2*f+1 : taille du patch
h = sigma;     % parametre de filtrage
acp = 9;
max = 0;

%%%%% ajout du bruit
Ibruitee = I + sigma*randn(n,m,'like',I);

%%%%% Variation du parametre de filtrage h
liste_h = [5:5:50]';
nlm_h = zeros(n,m,size(liste_h,1));
temps_h = zeros(size(liste_h));
psnr_h = zeros(size(liste_h));
ssim_h = zeros(size(liste_h));
for i = 1:size(liste_h,1)
    tic
    nlm_h(:,:,i) = nlm_final(Ibruitee,f,t,liste_h(i),acp,max);
    temps_h(i) = toc;
    psnr_h(i) = PSNR(I2,uint8(nlm_h(:,:,i)));
    ssim_h(i) = ssim(uint8(nlm_h(:,:,i)),I2);
end

%%%%% Variation de la fenetre de recherche t
liste_t = [5:15]';
nlm_t = zeros(n,m,size(liste_t,1));
temps_t = zeros(size(liste_t));
psnr_t = zeros(size(liste_t));
ssim_t = zeros(size(liste_t));
for i = 1:size(liste_t,1)
    tic
    nlm_t(:,:,i) = nlm_final(Ibruitee,f,liste_t(i),h,acp,max);
    temps_t(i) = toc;
    psnr_t(i) = PSNR(I2,uint8(nlm_t(:,:,i)));
    ssim_t(i) = ssim(uint8(nlm_t(:,:,i)),I2);
end
% %%%%% Sauvegarde des donnnees, changer le nom manuellement
% save('lena_acp3.mat','nlm_h','psnr_h','ssim_h');
% 
% %%%%% Chargement des donnnees
% load('lena_10.mat');i=1;
% load('lena_acp5.mat');i=2;
% load('lena_acp3.mat');i=3;
% 
% PSNR_H(:,i) = psnr_h;
% SSIM_H(:,i) = ssim_h;
% PSNR_F(:,i) = psnr_f;
% SSIM_F(:,i) = ssim_f;
% PSNR_T(:,i) = psnr_t;
% SSIM_T(:,i) = ssim_t;
%     
% % ACP à régler manuellement
% bar(liste_h,PSNR_H-28);xlabel('h');ylabel('psnr');title('psnr selon h / bruit = 10');legend({'lena','acp5','acp3'})
% set(gca, 'YTick', [0:1:6], 'YTickLabel',[0:1:6]+28)
% 
% bar(liste_h,SSIM_H-0.6);xlabel('h');ylabel('ssim');title('ssim selon h / bruit = 10');legend({'lena','acp5','acp3'})
% set(gca, 'YTick', [0:0.1:0.4], 'YTickLabel',[0:0.1:0.4]+0.6)
% 
% bar(liste_t*2+1,PSNR_T-32);xlabel('t');ylabel('psnr');title('psnr selon t / bruit = 10');legend({'lena','acp5','acp3'})
% set(gca, 'YTick', [0:0.3:1.5], 'YTickLabel',[0:0.3:1.5]+32)
% 
% bar(liste_t*2+1,SSIM_T-0.84);xlabel('t');ylabel('ssim');title('ssim selon t / bruit = 10');legend({'lena','acp5','acp3'})
% set(gca, 'YTick', [0:0.025:1], 'YTickLabel',[0:0.025:1]+0.84)


