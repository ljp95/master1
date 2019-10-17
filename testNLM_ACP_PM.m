%%%%%%%%%%%%%%%%%%%%%%%% Tests NLM + PM + ACP %%%%%%%%%%%%%%%%%%%%%%%%%%
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
acp = 9;
maxi = 0;

%%%%% Ajout du bruit gaussien
Ibruitee = I + sigma*randn(n,m,'like',I);
Ibruitee2 = uint8(Ibruitee);

%NLM voir nlm_final pour les commentaires
Ibruitee2 = reshape(Ibruitee,1,n*m);              
Inoisy3 = padarray(Ibruitee,[f,f],'symmetric');                         
patches = im2col(Inoisy3,[2*f+1,2*f+1],'sliding')'; 
filter = reshape(fspecial('gaussian',2*f+1,1),1,(2*f+1)^2);
patches = patches.*(filter.^(1/2));                 
indices = im2col(padarray(reshape(1:n*m,n,m),[t,t],'symmetric'),[2*t+1,2*t+1])'; 

%%%%% PM
nb_iter = 128;
NNF1 = zeros(n*m,nb_iter);
NNF2 = zeros(n*m,nb_iter);
for iter = 1:nb_iter
    NNF = patchmatch(Ibruitee,Ibruitee,f,2);
    NNF1(:,iter) = reshape(NNF(:,:,1),n*m,1);
    NNF2(:,iter) = reshape(NNF(:,:,2),n*m,1);
end

%Fusion des patchs
A = (NNF1+(NNF2-1)*m);
fen = (2*t+1).^2;
indices2 = zeros(n*m,fen+nb_iter);
indices2(:,1:fen) = indices;
indices2(:,fen+1:fen+nb_iter) = A;

%%%%% acp voir nlm_final pour commmentaires
moy = mean(patches);
patches = patches-moy;    
C = patches'*patches;
[u,s,v] = svd(C,0);
s = diag(s.^2);
v = v(:,1:acp)';
patches = (v*(patches'-moy'))';

% pondere tout les patchs voir nlm_final pour commentaires
h = sigma*sigma;
M = zeros(n*m,fen+nb_iter);                
for i=1:n*m
    M(i,:)=-sum((repmat(patches(i,:),size(indices2,2),1)-patches(indices2(i,:),:)).^2,2)';
end
M = exp(M/h);
S = sum(M.*Ibruitee2(indices2),2)./sum(M,2);

%%%%% Affichage NLM + PM + ACP
S = reshape(S,n,m);
S2 = uint8(S);
PSNR = psnr(I2,S2);
SSIM = ssim(I2,S2);
figure();imshow(S2);title(['NLM+PM+ACP',num2str(nb_iter/2),'/psnr : ',num2str(PSNR),'/ssim : ',num2str(SSIM)]);
