 classdef network_project_superclass2
    % superclass for all network project stuff. Not meant to be used directly.
    
    properties (Constant)
        % location of relevant folders
        Directory = ''
        % experiments run
        ExperimentFolder = [network_project_superclass2.Directory,'Experiments/']
        MappingFolder = [network_project_superclass2.Directory,'Mappings/']
        GroupFolder = [network_project_superclass2.Directory,'Groups/']
    end
    
    
    properties (SetAccess = immutable) % cannot be changed
        Date % date created
    end
    
    properties (SetAccess = protected)
        FullPathSaveName % same as above but with full path
    end
    
    
    
    methods
        
        % Construct object and save date created
        function obj = network_project_superclass2()
            obj.Date = datestr(now,'yy-mm-dd-HH-MM-SS');
        end
        
        
        % Save whichever subclass this is to a .mat file.
        function save(obj)
            if isempty(obj.Date)
                error('Cannot save, no date yet')
            end
            
            % create variable with correct name to store this object
            eval([obj.ObjName, '= obj;'])
            
            % save that variable in .mat file (or add it to existing mat file)
            filename = obj.FullPathSaveName;
            if exist(filename,'file') % add to file
                tmp = rmfield(load(filename), obj.ObjName);
                tmp.(obj.ObjName) = obj;
                % Resave, '-struct' flag tells MATLAB to store the fields as distinct variables
                save(filename, '-struct', 'tmp');
                
            else
                save(filename,obj.ObjName)
            end
        end
        
        % generates .mat filename without the entire path
        function name = matfilename(obj)
            name = obj.FullPathSaveName(length(obj.SaveFolder)+1:end);
        end
        
        % generates full filename path. Usable only when subclass
        % properties  have been defined (SaveFolder, ObjName, Name)
        function name = fullFilename(obj)
%             try
                name = [obj.SaveFolder,obj.ObjName,'_',obj.Name,'_',num2str(obj.Date),'.mat'];
%             catch
%                 error('Insufficient properties defined to create full file path.')
%             end
        end
        
        
        % Reload object from its saved location
        function obj = reload(obj)
            
            if nargout==0
                error('Loading does nothing if output is not specified!')
            end
            
            try
                filename = obj.FullPathSaveName;
            catch
                error('No specified file path.')
            end
            
            if exist(filename,'file')
                % load and store as obj, to be output.
                load(filename,obj.ObjName)
                eval(['obj = ',obj.ObjName,';'])
            else
                error('.mat file does not exist.')
            end
        end
    end
end
