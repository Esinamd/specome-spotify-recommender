import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useEffect, useState } from 'react';
import axios from 'axios';
import Recommending from './recommending';

const SpotifyConnect = () => {
  const navigate = useNavigate();
  let { state } = useLocation();
  console.log(state);

  //spotify auth request
  const client_id = '5dfac900abfa4581aad5e7428357e7a2';
  const redirect_uri = 'http://localhost:1212/SpotifyConnect';
  const auth_endpoint = 'https://accounts.spotify.com/authorize';
  const response_type = 'token';
  const scope = 'user-top-read';

  const [token, setToken] = useState('');
  const [artists, setArtists] = useState([]);
  const [tracks, setTracks] = useState([]);
  const [features, setFeatures] = useState([]);
  const [loggedIn, setLoggedIn] = useState(false);
  const [showTopList, setshowTopList] = useState(false);
  const [data, setData] = useState(null);
  const [clicked, setClicked] = useState(false);

  useEffect(() => {
    const hash = window.location.hash;
    let token = window.localStorage.getItem('token');

    //check if theres a hash or an already saved token in localstorage
    if (!token && hash) {
      token = hash
        .substring(1)
        .split('&')
        .find((elem) => elem.startsWith('access_token'))
        .split('=')[1];
      window.location.hash = '';
      window.localStorage.setItem('token', token);
    }
    console.log('acess token', token);
    setToken(token);
    if (token != null) {
      setLoggedIn(true);
    } else {
      setLoggedIn(false);
    }
  }, []);

  const logout = () => {
    //logout to remove hash
    setToken('');
    window.localStorage.removeItem('token');
    setLoggedIn(false);
  };

  const getArtists = async () => {
    try {
      const response = await axios.get(
        'https://api.spotify.com/v1/me/top/artists',
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
          params: {
            time_range: 'medium_term',
            limit: 10,
          },
        }
      );
      setArtists(response.data.items);
      getArtistsFeatures(response.data.items);
    } catch (error) {
      console.log(error);
      console.log('status', error.response.status);
      if (error.response.status == 401) {
        logout();
      }
    }
  };

  const getTracks = async () => {
    try {
      const response = await axios.get(
        'https://api.spotify.com/v1/me/top/tracks',
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
          params: {
            time_range: 'medium_term',
            limit: 10,
          },
        }
      );
      setTracks(response.data.items);
      getTracksFeatures(response.data.items);
    } catch {
      console.log(console.error);
    }
  };

  const getTracksFeatures = async (tracks: any) => {
    try {
      let ids = [];

      for (let i = 0; i < tracks.length; i++) {
        ids.push(tracks[i].id);
      }
      const id = ids.join(',');

      const response = await axios.get(
        'https://api.spotify.com/v1/audio-features',
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
          params: {
            ids: id,
          },
        }
      );
      setFeatures(response.data.audio_features);
      setshowTopList(true);
    } catch {
      console.log(console.error);
    }
  };

  const getArtistsFeatures = async (artists: any) => {
    try {
      let ids = [];

      for (let i = 0; i < artists.length; i++) {
        ids.push(artists[i].id);
      }
      const id = ids.join(',');

      const response = await axios.get('https://api.spotify.com/v1/artists', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
        params: {
          ids: id,
        },
      });
      setFeatures(response.data.artists);
      setshowTopList(true);
    } catch {
      console.log(console.error);
    }
  };

  const renderArtists = () => {
    return artists.map((artists) => (
      <div style={{ marginBottom: 10 }} key={artists.id}>
        {artists.name}
      </div>
    ));
  };

  const renderTracks = () => {
    return tracks.map((tracks) => (
      <div style={{ marginBottom: 10 }} key={tracks.id}>
        <div>
          {tracks.name} by{' '}
          {tracks.artists.map((artist) => artist.name).join(', ')}
        </div>
      </div>
    ));
  };

  const showTops = () => {
    console.log('preclick', clicked);
    if (state.type == 'songs' && loggedIn) {
      if (showTopList) {
        return (
          <div>
            <div className="renderedList">
              <h1>Here are your top tracks</h1>
              <div style={{ marginBottom: 30 }}>{renderTracks()}</div>
            </div>

            <div className="renderedButtons">
              {!clicked ? (
                <button className="centred" onClick={handleChange}>
                  Create your SpecoMe list!
                </button>
              ) : (
                showRecButton()
              )}
            </div>
          </div>
        );
      } else {
        return (
          <div>
            <h2>You're logged in to Spotify</h2>
            <button className="centred" onClick={getTracks}>
              Get Your Top Tracks!
            </button>
          </div>
        );
      }
    } else if (state.type == 'artists' && loggedIn) {
      console.log('clicked', clicked);
      if (showTopList) {
        return (
          <div>
            <div className="renderedList">
              <h1>Here are your top artists</h1>
              <div style={{ marginBottom: 30 }}> {renderArtists()}</div>
            </div>

            <div className="renderedButtons">
              {!clicked ? (
                <button className="centred" onClick={handleChange}>
                  Create your SpecoMe list!
                </button>
              ) : (
                showRecButton()
              )}
            </div>

            {}
          </div>
        );
      } else {
        return (
          <div>
            <h2>You're logged in to Spotify</h2>
            <button className="centred" onClick={getArtists}>
              Get Your Top Artists!
            </button>
          </div>
        );
      }
    }
  };

  const showRecButton = () => {
    console.log('data', data);
    if (data == (undefined || null) && clicked) {
      return <Recommending />;
    } else {
      return (
        <div>
          <h3>
            Want to see your SpecoMe recommendations based on your top{' '}
            {state.type}?
          </h3>
          {loggedIn && (
            <Link to="/SpotifyReturn" state={{ list: data, rType: state.type }}>
              <button style={{ width: 200, marginLeft: 180 }}>
                Let's See!
              </button>
            </Link>
          )}
        </div>
      );
    }
  };

  const submitList = async (data: any) => {
    try {
      const response = await axios.post(
        'http://127.0.0.1:5000/SpotifyConnect',
        data
      );
      setData(response.data);
      console.log('response', response.data);
    } catch (error) {
      console.error(error);
    }
  };

  const handleChange = () => {
    console.log('features', features);
    submitList({ features: features, type: state.type });
    setClicked(true);
  };

  return (
    <div>
      <div>
        <button className="back" onClick={() => navigate(-1)}>
          Back
        </button>
      </div>

      <div className="titleHeader menuMain">
        <h1 className="Title">SpecoMe</h1>
        <h3>A Spotify Recommendation App</h3>
      </div>

      {!token ? (
        <div className="spotifyLogin">
          <h2 className="subheadPos">Let's see your Spotify account!</h2>
          <button>
            <a
              href={`${auth_endpoint}?client_id=${client_id}&scope=${scope}&redirect_uri=${redirect_uri}&response_type=${response_type}&show_dialog=true`}
            >
              Connect with Spotify!
            </a>
          </button>
        </div>
      ) : (
        <div>
          <button className="spotifyLogout" onClick={logout}>
            Logout of Spotify!
          </button>
        </div>
      )}

      {showTops()}
    </div>
  );
};
export default SpotifyConnect;
