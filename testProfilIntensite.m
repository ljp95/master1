%%%%%%%%%%%%%%%%%%%%%%%% Profil d'intensite %%%%%%%%%%%%%%%%%%%%%%%
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

%%%%% Choix de la ligne/colonne
x = 38;
y = 285;

%%%%% affichage image + ligne/colonne de profil
J = I;
J(x,1:n) = 255; % ligne
% J(1:m,y) = 255; % colonne
figure();imagesc(J);colormap gray;title('profil sur la ligne blanche');

% Filtre moyenneur
filtre = fspecial('average',5);
Imean = conv2(Ibruitee,filtre,'same');
Imean2 = uint8(Imean);
% NLM
NLM = nlm_final(Ibruitee,f,t,h,acp,maxi);
NLM2 = uint8(NLM);
% Filtre gaussien
gauss = fspecial('gaussian',5,sigma);
Igauss = conv2(Ibruitee,gauss,'same');
Igauss2 = uint8(Igauss);
% Filtre median
Imedian = medfilt2(Ibruitee,[5,5]); 
Imedian2 = uint8(Imedian);

%%%%% Courbes sur une figure
subplot(2,3,1);plot(I(x,1:n));title('originale');
subplot(2,3,2);plot(Ibruitee(x,1:n));title('bruitee');
subplot(2,3,3);plot(Imean(x,1:n));title('moyenneur');
subplot(2,3,4);plot(NLM(x,1:n));title('NLM');
subplot(2,3,5);plot(Igauss(x,1:n));title('gaussien');
subplot(2,3,6);plot(Imedian(x,1:n));title('median');

%%%%% Nuage de points sur une figure
figure();
subplot(2,3,1);scatter(1:n,I(x,1:n),3);title('originale');
subplot(2,3,2);scatter(1:n,Ibruitee(x,1:n),3);title('bruitee');
subplot(2,3,3);scatter(1:n,Imean(x,1:n),3);title('moyenneur');
subplot(2,3,4);scatter(1:n,NLM(x,1:n),3);title('NLM');
subplot(2,3,5);scatter(1:n,Igauss(x,1:n),3);title('gaussien');
subplot(2,3,6);scatter(1:n,Imedian(x,1:n),3);title('median');

% %%%%% Courbes separees
% figure();plot(I(1:n,y),'r');title('originale');
% figure();plot(Ibruitee(1:n,y),'r');title('bruitee');
% figure();plot(Igauss(1:n,y),'r');title('gaussien');
% figure();plot(Imean(1:n,y),'r');title('moyenneur');
% figure();plot(NLM(1:n,y),'r');title('NLM');
% figure();plot(Imedian(1:n,y),'r');title('median');


%%%%% Comparaisons non bruitée, bruitéé et filtrage sur un même graphique
figure();plot(I(1:n,y),'b');
hold on;
plot(Ibruitee(1:n,y),'--g');plot(Imean(1:n,y),'r');
legend({'originale','bruitee)','moyenneur'},'Location','southwest')

% figure();plot(I(1:n,y),'b');
% hold on;
% plot(Ibruitee(1:n,y),'--g');plot(Igauss(1:n,y),'r');
% legend({'originale','bruitee)','gaussien'},'Location','southwest')
% 
% figure();plot(I(1:n,y),'b');
% hold on;
% plot(Ibruitee(1:n,y),'--g');plot(NLM(1:n,y),'r');
% legend({'originale','bruitee)','NLM'},'Location','southwest')
% 
% figure();plot(I(1:n,y),'b');
% hold on;
% plot(Ibruitee(1:n,y),'--g');plot(Imedian(1:n,y),'r');
% legend({'originale','bruitee)','median'},'Location','southwest')




