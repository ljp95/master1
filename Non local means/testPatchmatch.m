%%%%%%%%%%%%%%%%%%%%%%%%% Patchmatch %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   Initialisation des variables 
% I = imread('house2.png');
% I = imread('lena.gif');
I = imread('clown_lumi.bmp');
I2 = I;
I = double(I);
[n,m] = size(I);
sigma = 10; % parametre du bruit
t = 5;     % 2*t+1 : taille fenetre de recherche
f = 2;      % 2*f+1 : taille du patch
h = sigma;     % parametre de filtrage

%%%%% Ajout du bruit gaussien
Ibruitee = I + sigma*randn(n,m,'like',I);
Ibruitee2 = uint8(Ibruitee);

%%%%% Calcul NNF
f = 2;  % 2*f+1 : taille du patch
nb_iter = 5; % nombre d'iterations
NNF = patchmatch(I,I,f,nb_iter);

%%%%% affichage NNF
NNF1 = NNF(:,:,1);
NNF2 = NNF(:,:,2);
NNF3 = NNF(:,:,3);
figure();colormap gray; imagesc(NNF1);
figure();colormap gray; imagesc(NNF2);
figure();imagesc(NNF3);

%%%%% NLM avec les patches de Patchmatch
% on fait tourner nb_iter fois Patchmatch. A chaque fois on fait tourner 2 iterations de Patchmatch
nb_iter = 16;
NNF1 = zeros(n*m,nb_iter);
NNF2 = zeros(n*m,nb_iter);
for iter =1:nb_iter
    NNF = patchmatch(Ibruitee,Ibruitee,f,2);
    NNF1(:,iter) = reshape(NNF(:,:,1),n*m,1);
    NNF2(:,iter) = reshape(NNF(:,:,2),n*m,1);
end

% application des NLM avec les patches de patchmatch
% voir nlm_final pour commentaires
debut = tic;
Ibruitee2 = reshape(Ibruitee,n*m,1);
Ipad = padarray(Ibruitee,[f,f],'symmetric');
patches = im2col(Ipad,[2*f+1,2*f+1],'sliding')';
filter = reshape(fspecial('gaussian',2*f+1,1),1,(2*f+1)^2);
patches = patches.*(filter.^(1/2));           
M = zeros(n*m,nb_iter);                       
h = sigma*sigma;
for i=1:n*m
    M(i,:)=-sum((repmat(patches(i,:),nb_iter,1)-patches(NNF1(i,:)+(NNF2(i,:)-1)*m,:)).^2,2)';
end
M = exp(M/h);
NLM = sum(M.*Ibruitee2(NNF1+(NNF2-1)*m),2)./sum(M,2);
NLM = reshape(NLM,n,m);
NLM2 = uint8(NLM);
time = toc;

%%%%% affichage NLM avec patches de Patchmatch
PSNR = psnr(I2,NLM2);
SSIM = ssim(I2,NLM2);
figure();imshow(NLM2);title(['PM',num2str(nb_iter),'/Temps : ',num2str(debut-time),'/psnr : ',num2str(PSNR),'/ssim : ',num2str(SSIM)]);

