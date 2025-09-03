using Godot;
using System;
using System.Net.Sockets;
using System.Net;
using System.Text.Json;
using System.Text;
using System.Threading;
using System.Collections.Generic;
using System.IO;

public partial class SocketServer : Node2D
{
    [Export]
    public int StartPort = 12345; // Default, can be set per instance

    private TcpListener _server;
    private TcpClient _client;
    private NetworkStream stream;
    private bool _isRunning = true;

    public static Dictionary<int, string> LatestActions = new()
    {
        { 1, "idle" }
    };

    // Thread communication fields
    private string _pendingStateJson = null;
    private readonly object _stateLock = new();
    private AutoResetEvent _stateReadyEvent = new(false);
    private string _lastAction = null;

    public override void _Ready()
    {
        int port = StartPort;
        foreach (Node mainInstance in GetChildren())
        {
            var socketServer = mainInstance.GetNodeOrNull("SocketServer");
            if (socketServer != null)
            {
                socketServer.Set("StartPort", port);
                port++;
            }
        }
        Engine.TimeScale = 6.0f; // Ensure normal time scale
        _isRunning = true;
        new Thread(StartServer).Start();
    }

    // Graceful shutdown method
    public void ShutdownServer()
    {
        _isRunning = false;
        try
        {
            stream?.Close();
            _client?.Close();
            _server?.Stop();
        }
        catch (Exception ex)
        {
            GD.PrintErr("Error shutting down socket server: " + ex);
        }
    }

    private void StartServer()
    {
        try
        {
            _server = new TcpListener(IPAddress.Loopback, StartPort);
            _server.Start();
            GD.Print("Socket server started");

            try
            {
                _client = _server.AcceptTcpClient();
                stream = _client.GetStream();
            }
            catch (SocketException ex) when (!_isRunning && ex.SocketErrorCode == SocketError.Interrupted)
            {
                // Server was stopped intentionally, exit gracefully
                GD.Print("Socket server accept interrupted due to shutdown.");
                return;
            }
            catch (SocketException ex)
            {
                GD.PrintErr("Socket server accept error: " + ex);
                return;
            }

            while (_isRunning)
            {
                // Request state from main thread
                CallDeferred(nameof(PrepareAndSendState));

                // Wait for the state to be ready
                _stateReadyEvent.WaitOne();

                string json;
                lock (_stateLock)
                {
                    json = _pendingStateJson;
                    _pendingStateJson = null;
                }

                if (json == null)
                    continue; // No player found, skip

                // Send state to Python
                byte[] data = Encoding.UTF8.GetBytes(json);
                stream.Write(data, 0, data.Length);

                // Read action from Python
                byte[] buffer = new byte[128];
                int read = stream.Read(buffer, 0, buffer.Length);
                string action = Encoding.UTF8.GetString(buffer, 0, read).Trim();

                // Store action for player (on main thread)
                _lastAction = action;
                CallDeferred(nameof(SetLatestAction));
                Thread.Sleep(50);
            }
        }
        catch (SocketException ex) when (!_isRunning && ex.SocketErrorCode == SocketError.Interrupted)
        {
            // Server was stopped intentionally, exit gracefully
            GD.Print("Socket server interrupted due to shutdown.");
        }
        catch (Exception ex)
        {
            GD.PrintErr("Socket server error: " + ex);
        }
        finally
        {
            // Ensure sockets are closed on exit
            try
            {
                stream?.Close();
                _client?.Close();
                _server?.Stop();
            }
            catch (Exception ex)
            {
                GD.PrintErr("Error during socket server cleanup: " + ex);
            }
        }
    }

    private void PrepareAndSendState()
    {
        var mobStates = new List<Dictionary<string, object>>();
        foreach (var mobNode in GetTree().GetNodesInGroup("mobs"))
        {
            if (mobNode is Mushroom m)
                mobStates.Add(m.GetState());
        }

        // Collect XP drop states
        var xpDropStates = new List<Dictionary<string, float>>();
        foreach (var xpNode in GetTree().GetNodesInGroup("xpdrop"))
        {
            if (xpNode is Node2D xp)
            {
                xpDropStates.Add(new Dictionary<string, float>
            {
                { "x", xp.GlobalPosition.X },
                { "y", xp.GlobalPosition.Y }
            });
            }
        }

        Player player = GetTree().Root.FindChild("Player", true, false) as Player;
        if (player != null)
        {
            var state = new
            {
                x = player.GlobalPosition.X,
                y = player.GlobalPosition.Y,
                health = player.Health,
                timer = StopwatchLabel.GameTime,
                xpdrop = player._xp,
                mobs = mobStates,
                xp_drops = xpDropStates
            };
            string json = JsonSerializer.Serialize(state) + "\n";
            lock (_stateLock)
            {
                _pendingStateJson = json;
            }
        }
        else
        {
            lock (_stateLock)
            {
                _pendingStateJson = null;
            }
        }
        _stateReadyEvent.Set();
    }

    private void SetLatestAction()
    {
        Player player = GetTree().Root.FindChild("Player", true, false) as Player;
        if (player != null && _lastAction != null)
        {
            LatestActions[player.PlayerId] = _lastAction;
        }
    }
    public void SendStateToAI(Dictionary<string, object> state)
    {
        if (_client == null || stream == null || !_client.Connected)
        {
            GD.PrintErr("SocketServer: No client connected, cannot send state.");
            return;
        }

        try
        {
            // Serialize the state to JSON and add a newline
            string json = JsonSerializer.Serialize(state) + "\n";
            byte[] data = Encoding.UTF8.GetBytes(json);
            stream.Write(data, 0, data.Length);
            stream.Flush();
            GD.Print("SocketServer: Sent state to AI: " + json.Trim());
        }
        catch (Exception ex)
        {
            GD.PrintErr("SocketServer: Failed to send state to AI: " + ex);
        }
    }
}
