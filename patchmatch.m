function NNF = patchmatch(A,B,f,nb_iter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% renvoit NNF 
%   dimension 1 -> les x
% 	dimensoin 2 -> les y
%	dimension 3 -> distance entre patchs de l'image A et leur voisin en B
% A : premiere image
% B : seconde image
% f : patch de taille 2*f + 1
% max_iter : nombre d'iterations maximum
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Taille des images
[na,ma] = size(A);
[nb,mb] = size(B);

% Initialisation aleatoire
NNF(:,:,1) = randi([1,nb],na,ma);
NNF(:,:,2) = randi([1,mb],na,ma);
NNF(:,:,3) = inf(na,ma);

% Construction des patchs
Apad = padarray(A,[f,f]);
Bpad = padarray(B,[f,f]);
patchesA = im2col(Apad,[2*f+1,2*f+1],'sliding')';
patchesB = im2col(Bpad,[2*f+1,2*f+1],'sliding')';

% Construction radius : vecteur colonne 2^0 a 2^(maximum possible)
alpha = .5;
w = max(nb,mb)/2;
radius = round(w*alpha.^(0:-floor(log(w)/log(alpha))))';

disp(['Debut !']);
for iter = 1:nb_iter

%     Propagation de sens gauche-droite et haut-bas
    if mod(iter,2)==1
        disp(['Iteration ',num2str(iter),' ordre normal']);
        tic
        for i = 2:na
            for j = 2:ma
%                 [Haut, Centre, Gauche]
                X = [NNF(i-1,j,1),NNF(i,j,1),NNF(i,j-1,1)];
                Y = [NNF(i-1,j,2),NNF(i,j,2),NNF(i,j-1,2)];
                patch = patchesA(i+na*(j-1),:);
%                 Calcul des distances
                d1 = sum((patch-patchesB(X(1)+nb*(Y(1)-1),:)).^2);
                d2 = sum((patch-patchesB(X(2)+nb*(Y(2)-1),:)).^2);
                d3 = sum((patch-patchesB(X(3)+nb*(Y(3)-1),:)).^2);
                d = [d1,d2,d3];
                [NNF(i,j,3),indice] = min(d);
%                 Affectation du meilleur <=> argmin != centre
                switch indice
                    case 1
                        NNF(i,j,1) = min(NNF(i-1,j,1)+1,nb);
                        NNF(i,j,2) = NNF(i-1,j,2);
                    case 3
                        NNF(i,j,1) = NNF(i,j-1,1);
                        NNF(i,j,2) = min(NNF(i,j-1,2)+1,mb);
                end
%                 Recherche aleatoire
                R = -1+2*rand([size(radius,1),2]);
                U = [NNF(i,j,1),NNF(i,j,2)] + round(radius.*R);
                for k=1:size(U,2)
                    x = U(k,1);
                    y = U(k,2);
                    if and(and(x>=1,x<=nb),and(y>=1,y<=mb)) % Hors limite ou non des coordonnees
                        d = sum((patch-patchesB(x+nb*(y-1),:)).^2);
                        if d<NNF(i,j,3)
                            NNF(i,j,1) = x;
                            NNF(i,j,2) = y;
                            NNF(i,j,3) = d;
                        end
                    end
                end
            end
        end
        temps = toc

%                 Propagation de sens droite-gauche et bas-haut
    else
        disp(['Iteration ',num2str(iter),' ordre inverse']);
        tic
        for i = na-1:-1:1
            for j = ma-1:-1:1
%                 [Bas, Centre, Droite]
                X = [NNF(i+1,j,1),NNF(i,j,1),NNF(i,j+1,1)];
                Y = [NNF(i+1,j,2),NNF(i,j,2),NNF(i,j+1,2)];
                patch = patchesA(i+na*(j-1),:);
%                 Calcul des distances
                d1 = sum((patch-patchesB(X(1)+nb*(Y(1)-1),:)).^2);
                d2 = sum((patch-patchesB(X(2)+nb*(Y(2)-1),:)).^2);
                d3 = sum((patch-patchesB(X(3)+nb*(Y(3)-1),:)).^2);
                d = [d1,d2,d3];
                [NNF(i,j,3),indice] = min(d);
%                 Affectation du meilleur <=> argmin != centre
                switch indice
                    case 1
                        NNF(i,j,1) = max(1,NNF(i+1,j,1)-1);
                        NNF(i,j,2) = NNF(i+1,j,2);
                    case 3
                        NNF(i,j,1) = NNF(i,j+1,1);
                        NNF(i,j,2) = max(1,NNF(i,j+1,2)-1);
                end
%                 Recherche aleatoire
                R = -1+2*rand([size(radius,1),2]);
                U = [NNF(i,j,1),NNF(i,j,2)] + round(radius.*R);
                for k = 1:size(U,2)
                    x = U(k,1);
                    y = U(k,2);
                    if and(and(x>=1,x<=nb),and(y>=1,y<=mb))
                        d = sum((patch-patchesB(x+nb*(y-1),:)).^2); % Hors limite ou non des coordonnees
                        if d<NNF(i,j,3)
                            NNF(i,j,1) = x;
                            NNF(i,j,2) = y;
                            NNF(i,j,3) = d;
                        end
                    end
                end
            end
        end
        temps = toc
    end
end
disp(['Fin !']);
end