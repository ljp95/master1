%%%%%%%%%%%%%%%%%%%%%%%% Zoom sur les contours %%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%% Zones uniformes et textures %%%%%%%%%%%%%%%%%%%%%%%% 

%%%%% initialisation des variables
I = imread('lena.gif');
[n,m] = size(I);
I2 = I;
I = double(I);
sigma = 10; % parametre du bruit
f = 5;
t = 2;
h = sigma;
acp = 0;
maxi = 0;

%%%%% Ajout du bruit gaussien
Ibruitee = I + sigma*randn(n,m,'like',I);
Ibruitee2 = uint8(Ibruitee);

%%%%% application de tous les filtres
% NLM
NLM = nlm_final(Ibruitee,f,t,h,acp,maxi);
NLM2 = uint8(NLM);
% median
Imedian = medfilt2(Ibruitee,[3,3]);   
Imedian2 = uint8(Imedian);
% moyenneur
filtre = fspecial('average',5);
Imean = conv2(Ibruitee,filtre,'valid');
Imean2 = uint8(Imean);
% gaussien
filtre = fspecial('gaussian',10,3);
Igauss = conv2(Ibruitee,filtre,'valid');
Igauss2 = uint8(Igauss);

%%%%%%%%%% Contour %%%%%%%%%%
x = 120:200;
y = 30:110;
contour = I2(x,y);
contour_bruitee = Ibruitee(x,y);
contour_median = Imedian2(x,y);
contour_gauss = Igauss2(x,y);
contour_mean = Imean2(x,y);
contour_nlm = NLM2(x,y);

%%%%% affichage contour
figure()
subplot(2,3,1);imshow(contour);title('contour');
subplot(2,3,2);imshow(uint8(contour_bruitee));title('bruitee');
subplot(2,3,3);colormap gray;imshow(uint8(contour_median));title('median');
subplot(2,3,4);colormap gray;imshow(uint8(contour_gauss));title('gauss');
subplot(2,3,5);colormap gray;imshow(uint8(contour_mean));title('mean');
subplot(2,3,6);colormap gray;imshow(uint8(contour_nlm));title('NLM');

%%%%%%%%%% Texture %%%%%%%%%%
x = 70:150;
y = 160:240;
texture = I2(x,y);
texture_bruitee = Ibruitee(x,y);
texture_median = Imedian2(x,y);
texture_gauss = Igauss2(x,y);
texture_mean = Imean2(x,y);
texture_nlm = NLM2(x,y);

%%%%% affichage texture
figure();
subplot(2,3,1);imshow(texture);title('texture');
subplot(2,3,2);imshow(uint8(texture_bruitee));title('bruitee');
subplot(2,3,3);colormap gray;imshow(uint8(texture_median));title('median');
subplot(2,3,4);colormap gray;imshow(uint8(texture_gauss));title('gauss');
subplot(2,3,5);colormap gray;imshow(uint8(texture_mean));title('mean');
subplot(2,3,6);colormap gray;imshow(uint8(texture_nlm));title('NLM');

%%%%%%%%%% Homogene %%%%%%%%%%
%%%%% creation d'un 'echiquier' aleatoire
n = 512;
m = 512;
U = zeros(n,m);
pas = n/4;
for i=1:n/pas
    for j=1:m/pas
        U((i-1)*pas+1:i*pas,(j-1)*pas+1:j*pas) = randi(255,1);
    end
end
U2 = uint8(U);

%%%%% application de tous les filtres
Ubruitee = U + sigma*randn(size(U),'like',I);
Ubruitee2 = uint8(Ubruitee);
% NLM
UNLM = nlm_final(Ubruitee,f,t,h,acp,maxi);
UNLM2 = uint8(UNLM);
UNLM3= uint8(nlm_final(Ubruitee,f,t,20,acp,maxi));
UNLM4= uint8(nlm_final(Ubruitee,f,t,30,acp,maxi));
UNLM5= uint8(nlm_final(Ubruitee,f,t,40,acp,maxi));
% median
Umedian = medfilt2(Ubruitee,[3,3]);   
Umedian2 = uint8(Umedian);
% moyenneur
Umean = conv2(Ubruitee,filtre,'valid');
Umean2 = uint8(Umean);
% gaussien
Ugauss = conv2(Ubruitee,filtre,'valid');
Ugauss2 = uint8(Ugauss);

%%%%% calcul de la variance d'un carre
x = 1;
y = n/4-8 ;
variance = [var(Umedian(x:y,x:y),0,'all'),var(Ugauss(x:y,x:y),0,'all'),var(Umean(x:y,x:y),0,'all')...
,var(UNLM(x:y,x:y),0,'all'),var(double(UNLM2(x:y,x:y)),0,'all'),var(double(UNLM3(x:y,x:y)),0,'all')...
,var(double(UNLM4(x:y,x:y)),0,'all')];

%%%%% affichage filtrage + variance
figure()
subplot(2,3,1);imshow(U2);title('U');
subplot(2,3,2);imshow(Ubruitee2);title('bruitee');
subplot(2,3,3);colormap gray;imshow(Umedian2);title(['median, var = ',num2str(variance(1))]);
subplot(2,3,4);colormap gray;imshow(Ugauss2);title(['gauss, var = ',num2str(variance(2))]);
subplot(2,3,5);colormap gray;imshow(Umean2);title(['mean, var = ',num2str(variance(3))]);
subplot(2,3,6);colormap gray;imshow(UNLM2);title(['NLM h=10, var = ',num2str(variance(4))]);

%%%%% affichage NLM + variance
figure()
subplot(2,2,1);colormap gray;imshow(UNLM2);title(['NLM h=10, var = ',num2str(variance(4))]);
subplot(2,2,2);colormap gray;imshow(UNLM5);title(['NLM h=20, var = ',num2str(variance(5))]);
subplot(2,2,3);colormap gray;imshow(UNLM3);title(['NLM h=30, var = ',num2str(variance(6))]);
subplot(2,2,4);colormap gray;imshow(UNLM4);title(['NLM h=40, var = ',num2str(variance(7))]);

%%%%%%%%%% Poids %%%%%%%%%%
t = 40;
f = 2;
point = [160;65]; % contour1
% point = [43;435]; % contour2
% point = [110;200]; % texture
% point = [450;300]; % homogene

%%%%% voir nlm_final pour les commentaires
% image non bruitee
% patch = reshape(I(point(1)-f:point(1)+f,point(2)-f:point(2)+f),(2*f+1)^2,1);
% fenetre = I(point(1)-t-f:point(1)+t+f,point(2)-t-f:point(2)+t+f);

% image bruitee
patch = reshape(Ibruitee(point(1)-f:point(1)+f,point(2)-f:point(2)+f),(2*f+1)^2,1); 
fenetre = Ibruitee(point(1)-t-f:point(1)+t+f,point(2)-t-f:point(2)+t+f); 

% calculs
filtre = reshape(fspecial('gaussian',2*f+1,1),1,(2*f+1)^2); 
patch2 = repmat(patch,1,(2*t+1)^2)';
patches = im2col(fenetre,[2*f+1,2*f+1],'sliding')';
M = exp(-sum(((patch2-patches).^2).*filtre,2)/(h*h));

%%%%% affichage
% poids central à 1
J = reshape(M,2*t+1,2*t+1);
figure();colormap gray;
subplot(1,3,1);imagesc(uint8(fenetre));
subplot(1,3,2);imagesc(J);title({'poids central a 1'});
% poids central à max des autres poids
M(floor(size(M,1)/2)+1) = 0;
M(floor(size(M,1)/2)+1) = max(M);
J = reshape(M,2*t+1,2*t+1);
subplot(1,3,3);imagesc(J);title(['poids central a ',num2str(max(M))]);




