const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs/promises');

function createWindow() {
  const window = new BrowserWindow({
    width: 1440,
    height: 940,
    minWidth: 1120,
    minHeight: 720,
    backgroundColor: '#f6f7f9',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  window.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));
  if (process.env.WORKBENCH_SCREENSHOT) {
    window.webContents.once('did-finish-load', async () => {
      await new Promise(resolve => setTimeout(resolve, 1200));
      const image = await window.webContents.capturePage();
      await fs.writeFile(process.env.WORKBENCH_SCREENSHOT, image.toPNG());
      app.quit();
    });
  }
}

const dataPath = () => path.join(app.getPath('userData'), 'workbench-data.json');

ipcMain.handle('data:load', async () => {
  try { return JSON.parse(await fs.readFile(dataPath(), 'utf8')); }
  catch (error) { return error.code === 'ENOENT' ? null : Promise.reject(error); }
});

ipcMain.handle('data:save', async (_event, data) => {
  await fs.writeFile(dataPath(), JSON.stringify(data, null, 2));
  return true;
});

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => BrowserWindow.getAllWindows().length === 0 && createWindow());
});
app.on('window-all-closed', () => process.platform !== 'darwin' && app.quit());
