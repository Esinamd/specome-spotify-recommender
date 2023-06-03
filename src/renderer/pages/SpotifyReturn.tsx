import { Link, useLocation } from 'react-router-dom';

const closeApp = () => {
  window.electron.ipcRenderer.sendMessage('close', []);
};

const SpotifyReturn = () => {
  let { state } = useLocation();
  let showList: any = [];
  let cName = '';

  console.log('state', state);
  if (state.rType == 'songs') {
    cName = 'spotSongRes';
    let recList = [];
    for (let i = 0; i < state.list.length; i++) {
      state.list[i][1] = state.list[i][1].replace('[', '');
      state.list[i][1] = state.list[i][1].replace(']', '');
      while (state.list[i][1].includes("'")) {
        state.list[i][1] = state.list[i][1].replace("'", '');
      }
      const p = [state.list[i][0], state.list[i][1]];
      recList[i] = p;
    }

    showList = recList.map((item) => (
      <ul>
        {item[0]} by {item[1]}
      </ul>
    ));
  } else if (state.rType == 'artists') {
    cName = 'spotArtRes';
    showList = state.list.map((item: any) => <ul>{item}</ul>);
  }

  return (
    <div>
      <div className="titleHeader menuMain">
        <h1 className="Title">SpecoMe</h1>
        <h3>A Spotify Recommendation App</h3>
      </div>

      <h2 className={cName}>
        Based on your Spotify account, here is your SpecoMe list:
      </h2>
      <h3 className={cName}>{showList}</h3>
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
export default SpotifyReturn;
