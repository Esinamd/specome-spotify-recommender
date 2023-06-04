import { Link, useLocation } from 'react-router-dom';

const closeApp = () => {
  window.electron.ipcRenderer.sendMessage('close', []);
};

const RecSongs = () => {
  let { state } = useLocation();
  // console.log(state.listSongs);
  // console.log(state.listArtists);
  const playlist = [];

  for (let i = 0; i < state.listSongs.length; i++) {
    const p = [state.listSongs[i], state.listArtists[i]];
    playlist[i] = p;
  }

  console.log(playlist);

  const showPlaylist = playlist.map((track) => (
    <li style={{ marginBottom: 10 }}>
      {track[0]} by {track[1]}
    </li>
  ));

  return (
    <div>
      <div className="menuMain titleHeader">
        <h1 className="Title">SpecoMe</h1>
        <h3>A Spotify Recommendation App</h3>
      </div>

      <div className="recSongs">
        <h2>Based on the song '{state.song}', here's your SpecoMe playlist:</h2>
        <h3>{showPlaylist}</h3>
      </div>

      <div className="spotResButtons">
        <Link to="/">
          <button>Restart</button>
        </Link>
        <button onClick={closeApp}>Exit</button>
        {/* <button>Save Options</button> */}
      </div>
    </div>
  );
};
export default RecSongs;
