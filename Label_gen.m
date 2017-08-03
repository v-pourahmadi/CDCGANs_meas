
cd 'C:\Users\Vahid\Dropbox\Working_dir\Aut\Projects\__modleing_Channel\Channel_measure_jpg'
mat = dir('*.jpg'); 
for q = 1:length(mat) 
    name_split=strsplit(mat(q).name,'_');
    name_new=strcat(mat(q).name(1:end-3),'txt')
    
    if strcmp(name_split(3),'PedA')
        Velocity=str2num(name_split{4})+.5;
        fileID = fopen(name_new,'w');
        fprintf(fileID,'%i,%s\n', 1,Velocity);
        fclose(fileID);
    end
    if strcmp(name_split(3),'PedB')
        Velocity=str2num(name_split{4})+.5;
        fileID = fopen(name_new,'w');
        fprintf(fileID,'%i,%s\n', 2,Velocity);
        fclose(fileID);
    end
    if strcmp(name_split(3),'VehB')
        Velocity=str2num(name_split{4})+.5;
        fileID = fopen(name_new,'w');
        fprintf(fileID,'%i,%s\n', 3,Velocity);
        fclose(fileID);
    end

end 