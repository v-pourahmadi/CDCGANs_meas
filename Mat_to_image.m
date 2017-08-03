
cd 'C:\Users\Vahid\Dropbox\Working_dir\Aut\Projects\__modleing_Channel\Channel_measure'
mat = dir('*.mat'); 
for q = 1:length(mat) 
    data = load(mat(q).name); 
    name_new=strcat(mat(q).name(1:end-3),'jpg')
    %data_2d(:,:,1)=abs(data.H_to_save)./max(max(abs(data.H_to_save)));
    tmp=abs(data.H_to_save);
    tmp_norm=(tmp-min(min(tmp)))./(max(max(tmp))-min(min(tmp)));
    data_2d(:,:,1)=tmp_norm;

    %data_2d(:,:,2)=angle(data.H_to_save)./max(max(angle(data.H_to_save)));
    tmp=angle(data.H_to_save);
    tmp_norm=(tmp-min(min(tmp)))./(max(max(tmp))-min(min(tmp)));
    data_2d(:,:,2)=tmp_norm;
    
    data_2d(:,:,3)=0*angle(data.H_to_save);
    
    imwrite(data_2d,strcat('C:\Users\Vahid\Dropbox\Working_dir\Aut\Projects\__modleing_Channel\Channel_measure_jpg\',name_new))
end 