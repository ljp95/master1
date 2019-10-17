function NLM = nlm_final(Ibruitee,f,t,h,acp,maxi)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Renvoit NLM en vectorisant les donnees
% Ibruitee : image bruitee
% f : fenetre de recherche de taille 2*f+1
% t : patch de taille 2*t+1
% h : parametre de filtrage
% acp : nombre de composantes principales a garder
% max : booleen : -> 1 si poids central = max des autres poids
%                 -> 0 si poids central = 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%% Initialisation des variables %%%%%%%%%%%%%%%%
h = h*h;
[n,m] = size(Ibruitee);
% Reshape Ibruitee en vecteur (1,n*m)
Inoisy2 = reshape(Ibruitee,1,n*m);              
% Padding Ibruitee (n*m+f,n*m+f)
Inoisy3 = padarray(Ibruitee,[f,f],'symmetric');   
% Preallocation matrice de distances entre un patch et ses patchs voisins (n*m,(2*t+1)^2)
M = zeros(n*m,(2*t+1)^2);                       
% Tous les patchs sous forme de vecteurs (n*m,(2*f+1)^2)
patches = im2col(Inoisy3,[2*f+1,2*f+1],'sliding')'; 
% filtre gaussien en vecteur (1,(2*f+1)^2)
filter = reshape(fspecial('gaussian',2*f+1,1),1,(2*f+1)^2);
% Application filtre gaussien sur tous les patchs, prise en compte du carré
patches = patches.*(filter.^(1/2));                 
% indices des patchs voisins pour chaque patch(n*m,(2*t+1)^2)
indices = im2col(padarray(reshape(1:n*m,n,m),[t,t],'symmetric'),[2*t+1,2*t+1]); 
%%%%% ACP
if acp
    moy = mean(patches);
    patches = patches-moy;    
    C = patches'*patches;
    [u,s,v] = svd(C,0);
    s = diag(s.^2);
    v = v(:,1:acp)';
    patches = (v*(patches'-moy'))';
end

%%%%%%%%%%%%%%%% Calcul %%%%%%%%%%%%%%%%
for i=1:n*m
%     Euclidienne
    M(i,:)=-sum((repmat(patches(i,:),size(indices,1),1)-patches(indices(:,i),:)).^2,2)';
%     Manhattan
%     M(i,:)=-sum(abs(repmat(patches(i,:),size(indices,1),1)-patches(indices(:,i),:)),2)';
end
M = exp(M/h);

%Max au lieu de 1
if maxi 
    M(:,floor(size(M,2)/2)+1) = 0;
    M(:,floor(size(M,2)/2)+1) = max(M');
end
NLM = sum(M.*Inoisy2(indices)',2)./sum(M,2);
NLM = reshape(NLM,n,m);
end


