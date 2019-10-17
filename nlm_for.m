function NLM = nlm_for(Inoisy,f,t,h)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Renvoit NLM en utilisant de simples boucle for
% Ibruitee : image bruitee
% f : fenetre de recherche de taille 2*f+1
% t : patch de taille 2*t+1
% h : parametre de filtrage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%% Initialisation des variables %%%%%%%%%%%%
[n,m] = size(Inoisy);
NLM = zeros(n,m);
Inoisy2 = padarray(Inoisy,[t+f, t+f],'symmetric');
filtre = fspecial('gaussian',2*f+1,1);

%%%%%%%%%%%% Calcul %%%%%%%%%%%%
for j=1:n
     for i=1:m
        i2 = i+t+f;
        j2 = j+t+f;
        A = Inoisy2(i2-f:i2+f,j2-f:j2+f);                   % patch pour chaque pixels
        Z = 0;                                              % pour normalisation
        for jj = j2-t:j2+t
            for ii = i2-t:i2+t
%                 patch pour chaque voisins de A
                B = Inoisy2(ii-f:ii+f,jj-f:jj+f);           
%                 Euclidienne
                w = exp(-sum(sum(filtre.*((A-B).^2)))/(h*h));
%                 Manhattan
%                 w = exp(-sum(sum(filter.*abs(A-B)))/(h*h));
                Z = Z+w;                                
                NLM(i,j) = NLM(i,j) + w*Inoisy2(ii,jj);   
            end
        end
        NLM(i,j) = NLM(i,j)/Z;
    end
end