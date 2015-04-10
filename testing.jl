using DataFrames

function read_data() 
    data = readtable("train.csv")
    users = readtable("profiles.csv")
    artists = readtable("artists.csv")
    return data, users, artists
end

function get_maps1(data) 
    n_users = size(users)[1]
    n_artists = size(artists)[1]
    user_to_int = {"" => 0}
    artist_to_int = {"" => 0}

    count = 1
    for user in users[1]
        user_to_int[user] = count
        count += 1
    end

    count = 1
    for artist in artists[1]
        artist_to_int[artist] = count
        count += 1
    end
    return user_to_int, artist_to_int
end

function get_maps2(data)   
    n_data = size(data)[1]
    means = {"user" => 0}
    vars = {"user" => 0}
    usersongs = {"user" => []}
    userartists = {"user" => []}
    artistusers = {"artist" => []}
    
    println("Building usersongs, userartists, and artistusers")
    for n in 1:n_data
        user = data[n,:user]
        artist = data[n,:artist]
        plays = data[n,:plays]
        if plays > 0
            if haskey(usersongs, user)
                push!(usersongs[user],plays)
            else
                usersongs[user] = [plays]
            end
        end

        if haskey(userartists, user)
            push!(userartists[user],artist)
        else
            userartists[user] = [artist]
        end
        
        if haskey(artistusers, artist)
            push!(artistusers[artist],user)
        else
            artistusers[artist] = [user] 
        end
        if n % 500000 == 0
            println("Processed ",n, " data points.")
        end
    end
    println("Done.")
    println("Building means and vars")
    for user in data[:user]
        means[user] = mean(usersongs[user])
        vars[user] = var(usersongs[user])
        if (vars[user] < 0.000000001)
            vars[user] = 1 # TODO: pesky divide by 0 errors...
        end
    end
    println("Done.")
    return usersongs, userartists, means,vars,artistusers
end

function standardize_data(data) # NOT in place standardization
    n_data = size(data)[1]
    stdplays = zeros(n_data)
    for n in 1:n_data
        user = data[n,:user]
        cur = data[n,:plays]
        stdplays[n] = (cur - means[user])/sqrt(vars[user])+2
    end
    dfB = DataFrame(std_plays=stdplays)
    stddata = hcat(data, dfB)
    return stddata
end

function standardize_usersongs(usersongs)
    std_usersongs = {"user" => []}
    for user in keys(usersongs)
        s = size(usersongs[user])[1]
        std_usersongs[user] = zeros(s)
        
        for i in 1:s
            std_usersongs[user][i] = 2+(usersongs[user][i] - means[user])/sqrt(vars[user])
        end
    end
    return std_usersongs
end

function read_test()
    test = readtable("test.csv")
    return test
end
function compute_distance(user_u,user_v,usersongs,std_usersongs,userartists,means,vars)
    artists_u = userartists[user_u]
    artists_v = userartists[user_v]
    inter = intersect(artists_u,artists_v)
    len_inter = size(inter)[1]
    
    if len_inter <= 0
        return 0
    else
        vec_u = zeros(len_inter)
        vec_v = zeros(len_inter)
        for i in 1:len_inter
            artist = inter[i]
            i_u = findfirst(userartists[user_u], artist)
            i_v = findfirst(userartists[user_v], artist)
            vec_u[i] = (std_usersongs[user_u][i_u])
            vec_v[i] = (std_usersongs[user_v][i_v])
            #vec_u[i] = (usersongs[user_u][i_u] - means[user_u])/sqrt(vars[user_u])+2
            #vec_v[i] = (usersongs[user_v][i_v] - means[user_v])/sqrt(vars[user_v])+2
        end
        #cos_dist = -dot(vec_u, vec_v)/(vecnorm(vec_u)*vecnorm(vec_v))
        cos_dist = vecnorm(vec_u - vec_v)
        return cos_dist
    end
end

function make_predictions(test, artistusers, distances, usersongs,std_usersongs,userartists,means,vars)
    println("Making predictions.")
    n_test = size(test)[1]
    predictions = zeros(n_test)
    ids = zeros(n_test)
    tic()
    for t in 1:n_test
        user_t = test[t,:user]
        artist_t = test[t,:artist]
        
        common_users = artistusers[artist_t]
        #if user_t in common_users
        #    new_usrs = zeros( size(common_users)[1] - 1 )
        #    for (user in common_users)
        #        z = 1
        #        if user != user_t
        #            new_usrs[z] = usr
        #            z += 1
        #        end
        #    end
        #    common_users = new_usrs
        #end
        
        cur_dists = zeros( size(common_users)[1] )
        
        
        count = 1
        for user_c in common_users
            newkey = (user_t, user_c)
            newkey2 = (user_c, user_t)
            if !haskey(distances, newkey)
                # recompute distance
                dist = compute_distance(user_t, user_c,usersongs,std_usersongs,userartists,means,vars)
                distances[newkey] = dist
                distances[newkey2] = dist
                cur_dists[count] = dist
            else
                cur_dists[count] = distances[newkey]
            end
            #if user_c == user_t
            #    cur_dists[count] = 100000 
            #end
            count += 1
        end
        
        # get your smallest distances
        p = sortperm(cur_dists)
        
        # nearest
        nearest = min(10, size(common_users)[1])
        
        plays_list = zeros(nearest)
        
        for i in 1:nearest
            # get common user
            user_c = common_users[p[i]]
            # index of artist in userartists list
            i_c = findfirst(userartists[user_c], artist_t)
            #plays = usersongs[user_c][i_c]
            plays_list[i] = std_usersongs[user_c][i_c]-2 #(plays - means[user_c])/sqrt(vars[user_c])
        end
        predictions[t] = median(plays_list)*sqrt(vars[user_t]) + means[user_t]
        if predictions[t] < 0
            predictions[t] = 1
        end
        ids[t] = test[t,:Id]
        
        if t % 100 == 0
            df = DataFrame(A=ids, B=predictions)
            writecsv("out.csv",hcat(ids,predictions)) 
        end
        if t % 100000 == 0
            println("Predicted ", t, " values. Took ", toc())
            tic()
        end
    end
    println("Done predicting!")
    return ids,predictions
end


start = 2000000
data, users, artists = read_data()
test = read_test()
users_to_int, artists_to_int = get_maps1(data)
usersongs, userartists, means,vars,artistusers = get_maps2(data)
std_usersongs = standardize_usersongs(usersongs)
distances2 = {("a","b")=>0}
ids,predict = make_predictions(test[start:,:], artistusers, distances2, usersongs,std_usersongs,userartists,means,vars)