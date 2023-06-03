import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useState } from 'react';
import Recommending from './recommending';

const PageSongs2 = () => {
  const navigate = useNavigate();

  const [data, setData] = useState({});
  const [name, setName] = useState('');
  const [artist, setArtist] = useState('');
  const [no, setNo] = useState('');
  const [songListSongs, setSongListSongs] = useState();
  const [songListArtists, setSongListArtists] = useState();
  const [clicked, setClicked] = useState(false);

  const ShowRecButton = (listedBool) => {
    if (listedBool.listedData == true) {
      return (
        <Link
          to="/RecSongs"
          state={{
            song: name,
            listSongs: songListSongs,
            listArtists: songListArtists,
          }}
        >
          <button>Reveal Recommended Artists</button>
        </Link>
      );
    } else if (listedBool.listedData == false) {
      setClicked(false);
      return (
        <h2>
          Oh no! There seems to be an error, please check your inputs or try
          another song
        </h2>
      );
    } else if (listedBool.listedData == undefined && clicked) {
      return <Recommending />;
    }
  };

  const submitSong = async (data: any) => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/Songs2', data);
      setData(response.data);
      setSongListSongs(response.data.listSongs);
      setSongListArtists(response.data.listArtists);
    } catch (error) {
      console.error(error);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    console.log(name, artist, no);
    setData({});
    submitSong({ name: name, artist: artist, number: no });
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

      <form onSubmit={handleChange}>
        <label for="song">
          <h2>What's the song name?</h2>
        </label>
        <input
          type="text"
          id="song"
          placeholder="My favourite song..."
          onChange={(e) => setName(e.target.value)}
        ></input>

        <label for="artist">
          <h2>Who is it by?</h2>
        </label>
        <input
          type="text"
          id="artist"
          placeholder="My favourite artist..."
          onChange={(e) => setArtist(e.target.value)}
        ></input>

        <label for="number">
          <h2>Number of songs to generate?</h2>
        </label>
        <input
          type="text"
          id="number"
          placeholder="e.g. 10"
          onChange={(e) => setNo(e.target.value)}
        ></input>

        <div className="formButton">
          {!clicked && <button type="submit">Recommend!</button>}
        </div>

        <div>
          <ShowRecButton listedData={data.listed} />
        </div>
      </form>
    </div>
  );
};
export default PageSongs2;
