import { Link, useLocation } from 'react-router-dom';

const closeApp = () => {
  window.electron.ipcRenderer.sendMessage('close', []);
};

const RecArtists = () => {
  let { state } = useLocation();

  console.log(state);
  const showPlaylist = state.list.map((artist: any) => <ul>{artist}</ul>);

  return (
    <div>
      <div className="menuMain titleHeader">
        <h1 className="Title">SpecoMe</h1>
        <h3>A Spotify Recommendation App</h3>
      </div>

      <div className="recArtists">
        <h2>
          Based on {state.artist}, here is your SpecoMe list of similar artists:
        </h2>
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
export default RecArtists;
